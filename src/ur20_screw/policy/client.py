"""Focused Responses API policy and tool surface for the UR20 screw task."""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from typing import Any

import httpx
import numpy as np
from PIL import Image

from ur20_screw.contracts import Observation

POLICY_CAMERAS = ("agentview", "wristview")
MAX_LIVE_POLICY_IMAGES = len(POLICY_CAMERAS) * 2


def _image_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False,
        },
        "strict": False,
    }


class UR20Policy:
    def __init__(self, *, model: str, api_key: str, endpoint: str,
                 transport: httpx.BaseTransport, timeout: float, max_calls: int,
                 max_output_tokens: int, prior_learnings: str | None = None,
                 effort: str | None = None, history_keep_observations: int = 4,
                 prompt_cache_key: str | None = None,
                 prompt_cache_retention: str | None = None,
                 adaptive_scheduler: bool = True,
                 adaptive_observation: bool = True):
        self.model = model
        self.endpoint = endpoint
        self.max_calls = max_calls
        self.max_output_tokens = max_output_tokens
        self.effort = effort
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention
        self.adaptive_scheduler = bool(adaptive_scheduler)
        self.adaptive_observation = bool(adaptive_observation)
        self.calls = 0
        self.max_response_retries = 2
        self.history_keep_observations = max(1, int(history_keep_observations))
        self.history_compactions = 0
        self.adaptive_single_view_observations = 0
        self.http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, transport=transport,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self.input: list[dict[str, Any]] = []
        self.system = (
            "You control the UR20 battery screw-removal simulation through one tool call per turn. "
            "Use the robot state and labeled RGB views, re-check after each motion, and do not claim "
            "success without physical evidence. Before axial contact, when vision confidence is at least "
            "0.7 and clearance is visible, prefer approach_until_contact for the clear pre-contact descent, "
            + ("or move_sequence with 2-8 bounded Cartesian targets for lateral alignment. The local controller "
               "can then commit a safe action segment without an LLM round trip per point. "
               "Use move_to for uncertain alignment, and stop using move_sequence as soon as contact, force, "
               "or low confidence appears. " if self.adaptive_scheduler else
               "Use move_to for one bounded Cartesian target at a time and re-check after every motion. ")
            + "The task instruction and robot operating notes follow.\n\n"
        )
        if prior_learnings:
            self.system += "Prior task notes:\n" + prior_learnings + "\n\n"
        self.tools = [
            _tool("move_to", "Move the tool tip to absolute world-frame x, y, z targets in meters. "
                  "Unnamed coordinates stay fixed. Changes are limited by the controller. Include a concise observation-based note.",
                  {"targets": {"type": "object", "properties": {
                      axis: {"type": "number"} for axis in ("x", "y", "z")},
                      "additionalProperties": False},
                   "note": {"type": "string"}}, ["targets", "note"]),
            _tool("move_sequence", "Execute a short preplanned sequence of absolute Cartesian targets. "
                  "Use only for visible-clearance approach/alignment moves before axial contact; each "
                  "point is checked by the controller and the sequence stops on contact or a safety gate. "
                  "Do not include spiral or extraction actions.",
                  {"targets": {"type": "array", "items": {"type": "object", "properties": {
                      axis: {"type": "number"} for axis in ("x", "y", "z")},
                      "additionalProperties": False}, "minItems": 2, "maxItems": 8},
                   "note": {"type": "string"}}, ["targets", "note"]),
            _tool("approach_until_contact", "Hand the clear, pre-contact approach to the local compliant controller. "
                  "It uses the latest vision estimate, descends in bounded increments, and stops immediately "
                  "on axial contact, low-confidence safety hold, or the approach limit. Use only before contact; "
                  "after contact call spiral_search.",
                  {"note": {"type": "string"}}, ["note"]),
            _tool("done", "End the rollout when the task is complete.",
                  {"summary": {"type": "string"}}, ["summary"]),
        ]
        if not self.adaptive_scheduler:
            self.tools = [tool for tool in self.tools
                          if tool["name"] not in {"move_sequence", "approach_until_contact"}]

    def enable_spiral_search(self) -> None:
        self.tools.append(_tool(
            "spiral_search", "After detected axial contact, start the blocking force-controlled socket seating search.",
            {"note": {"type": "string"}}, ["note"]))

    def enable_socket_turn(self) -> None:
        self.tools.append(_tool(
            "turn_socket", "After stable tactile seating, run the bounded threaded screw extraction operation.",
            {"note": {"type": "string"}}, ["note"]))

    def observe(self, observation: Observation, robot_docs: str) -> None:
        if not self.input:
            self.system += "Robot operating notes:\n" + robot_docs
            self.input.append({"role": "system", "content": self.system})
        parts: list[dict[str, Any]] = [{"type": "input_text", "text": (
            f"Goal: {observation.instruction}\nCurrent state (JSON): "
            + json.dumps({key: np.asarray(value).tolist() for key, value in observation.state.items()})
        )}]
        force = np.asarray(observation.state.get("force_torque_compensated", np.zeros(6)), dtype=float)
        confidence = float(np.asarray(observation.state.get("vision_confidence", [0.0])).reshape(-1)[0])
        xy_error = float(np.linalg.norm(np.asarray(
            observation.state.get("vision_xy_error_m", [99.0, 99.0]), dtype=float)))
        locked_clearance = confidence >= 0.7 and xy_error <= 0.002 and abs(float(force[2])) < 1.0
        camera_names = (("wristview",) if locked_clearance else POLICY_CAMERAS) \
            if self.adaptive_observation else POLICY_CAMERAS
        if locked_clearance and self.adaptive_observation:
            self.adaptive_single_view_observations += 1
        for name in camera_names:
            image = observation.images.get(name)
            if image is None:
                continue
            parts.append({"type": "input_text", "text": f"Camera: {name}"})
            parts.append({"type": "input_image", "image_url": _image_url(image)})
        self.input.append({"role": "user", "content": parts})
        image_messages = [
            item for item in self.input
            if item.get("role") == "user" and isinstance(item.get("content"), list)
            and any(part.get("type") == "input_image" for part in item["content"])
        ]
        live_images = sum(
            part.get("type") == "input_image"
            for item in image_messages
            for part in item["content"]
        )
        for item in image_messages:
            if live_images <= MAX_LIVE_POLICY_IMAGES:
                break
            removed = sum(part.get("type") == "input_image" for part in item["content"])
            item["content"] = [
                part for part in item["content"] if part.get("type") != "input_image"
            ]
            item["content"].append({"type": "input_text", "text": "[Earlier camera images omitted]"})
            live_images -= removed
        if self.adaptive_scheduler:
            self._compact_history()

    def _compact_history(self) -> None:
        """Keep recent observation turns while preserving the Responses call structure."""
        observations = [
            item for item in self.input
            if item.get("role") == "user" and isinstance(item.get("content"), list)
        ]
        excess = len(observations) - self.history_keep_observations
        if excess <= 0:
            return
        for item in observations[:excess]:
            item["content"] = [{
                "type": "input_text",
                "text": "[Earlier observation compacted; use the latest observation for control.]",
            }]
        self.history_compactions += excess

    def act(self) -> tuple[str, dict[str, Any], str]:
        if self.calls >= self.max_calls:
            raise RuntimeError("LLM call limit reached")
        payload: dict[str, Any] | None = None
        for attempt in range(self.max_response_retries + 1):
            response = self.http.post(
                self.endpoint,
                json={"model": self.model, "input": self.input, "tools": self.tools,
                      "max_output_tokens": self.max_output_tokens, "store": False,
                      "parallel_tool_calls": False,
                      "tool_choice": "required",
                      **({"prompt_cache_key": self.prompt_cache_key}
                         if self.prompt_cache_key else {}),
                      **({"prompt_cache_retention": self.prompt_cache_retention}
                         if self.prompt_cache_retention else {}),
                      **({"reasoning": {"effort": self.effort}} if self.effort else {})},
            )
            if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504, 524}:
                if attempt == self.max_response_retries:
                    response.raise_for_status()
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            try:
                payload = response.json()
                break
            except ValueError:
                # 网关偶发返回 200 + 空/非 JSON 报文；重发同一请求，预算仍由
                # BudgetTransport 逐次记账。
                if attempt == self.max_response_retries:
                    raise
                time.sleep(2.0 * (attempt + 1))
        assert payload is not None
        output = payload.get("output", [])
        self.input.extend(output)
        calls = [item for item in output if item.get("type") == "function_call"]
        self.calls += 1
        if len(calls) != 1:
            raise RuntimeError(f"expected one tool call, received {len(calls)}")
        call = calls[0]
        arguments = json.loads(call.get("arguments") or "{}")
        return str(call["name"]), arguments, str(call["call_id"])

    def tool_result(self, call_id: str, result: str) -> None:
        self.input.append({"type": "function_call_output", "call_id": call_id, "output": result})

    def close(self) -> None:
        self.http.close()
