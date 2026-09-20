O2C_PROMPT = """Based on the navigation history and current 4-directional views, decide the next direction.
Notice: The candidate waypoints in the images are marked with colored dots and white ID numbers. The image size is {width}x{height} pixels.

### Waypoint Color Guide
- **Cyan dots**: Candidate waypoints - potential directions you can explore NOW. These are your primary navigation options.
- **Red dots**: Historical waypoints - locations you have ALREADY VISITED. AVOID selecting these unless absolutely necessary (e.g., backtracking due to dead end).

### Understanding 'History'
'History' represents the places you have already explored along with their corresponding images. It includes both correct movements according to the 'Instruction' and some past mistaken explorations.
**You must use 'History' to verify whether an instruction step has already been completed.** Seeing an object in an image does **not** mean you have not yet passed it; instead, confirm whether the object was previously observed **from a past location** before assuming that step remains incomplete.

### Analysis Requirement
For each provided image, **analyze it in conjunction with 'Instruction' and 'History'** to determine:
1. What **parts of the instruction have already been executed**?
2. What **steps remain to be executed**?
3. Whether your **current position is still aligned with the instruction** or if you have deviated.
Your reasoning must be based on **actual past movements, not just object visibility in images** to help you make decision.

Choose ONE of these actions:
1. stop - ONLY if you have actually REACHED the target. If you see the target but it is still several meters away, do NOT stop; instead, select its ID to navigate closer.
2. navigate to forward - continue straight ahead
3. navigate to left - turn left and go forward
4. navigate to right - turn right and go forward
5. navigate to behind - turn around and go forward

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen action. Explicitly state why you chose this direction.>",
    "action": "stop" or "navigate to forward|left|right|behind",
    "target_id": <the integer ID of a point in your chosen direction, or null if there are no marked points>,
    "navigable_bbox": [<xmin>, <ymin>, <xmax>, <ymax>] or null
}}

Guidelines:
- WAYPOINT PRIORITY: You MUST strongly prioritize navigating in directions that contain visible CYAN dots (candidate waypoints). Try your best to avoid directions with completely zero marked points or directions blocked by walls.
- AVOID BACKTRACKING: RED dots mark the directions you came from. Do NOT select them under normal circumstances — going backwards wastes time and may cause navigation loops. Only consider a RED dot direction when: (1) you realize you took a wrong turn and need to go back to a previous junction to retry, or (2) ALL cyan directions are dead ends and you have no other way forward.
- STOPPING RULE: You must be right next to the target to stop. If the target is visible but small or far away, you MUST continue to navigate towards it by selecting its point ID.
- IF NAVIGATING WITH POINTS: If there are ANY marked points in your chosen direction, you MUST select the ID of the most relevant one and set "navigable_bbox" to null.
- IF NAVIGATING WITHOUT POINTS: If your chosen direction has NO marked points but is the only correct path, you MUST provide a "navigable_bbox" [xmin, ymin, xmax, ymax] (in pixels) framing a safe, navigable area (like an open hallway or floor space). The system will use the bottom-center of this box as the target point. Set "target_id" to null.
- Choose the direction that best advances toward the goal.
- BBOX BOUNDS: If using "navigable_bbox", coordinates MUST strictly fall within the image resolution (0 <= xmin < xmax <= {width}, 0 <= ymin < ymax <= {height}). Do NOT output coordinates beyond these limits.
"""
