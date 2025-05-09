import json
import html
from typing import List

from .logging import logger
from .definitions import FSMInstance, State


class PromptBuilder:
    """
    Builder for creating prompts for the LLM using semantic XML notation.
    """

    def build_system_prompt(self, instance: FSMInstance, state: State) -> str:
        """
        Build a system prompt for the current state with clearer instructions about valid transitions.

        Args:
            instance: The FSM instance
            state: The current state

        Returns:
            A system prompt string using semantic XML notation
        """
        logger.debug(f"Building system prompt for state: {state.id}")

        # Pre-process data and escape values
        fsm_id = html.escape(instance.fsm_id)
        state_id = html.escape(state.id)
        description = html.escape(state.description)
        purpose = html.escape(state.purpose)

        # Get available states for transitions
        available_states = [t.target_state for t in state.transitions]
        available_states_str = ", ".join([f"'{html.escape(s)}'" for s in available_states])

        # Create XML structure
        xml_parts = [
            "<fsm>",
            f"  <metadata>",
            f"    <name>{fsm_id}</name>",
            f"    <currentState>{state_id}</currentState>",
            f"  </metadata>",
            f"",
            f"  <stateInfo>",
            f"    <description>{description}</description>",
            f"    <purpose>{purpose}</purpose>"
        ]

        # Add instructions if available
        if state.instructions:
            xml_parts.append(f"    <instructions>{html.escape(state.instructions)}</instructions>")

        xml_parts.append("  </stateInfo>")

        # Add required context keys
        if state.required_context_keys:
            xml_parts.append("  <dataCollection>")
            xml_parts.append("    <requiredFields>")
            for key in state.required_context_keys:
                xml_parts.append(f"      <field>{html.escape(key)}</field>")
            xml_parts.append("    </requiredFields>")
            xml_parts.append("  </dataCollection>")

        # Add available transitions
        xml_parts.append("  <transitions>")

        if state.transitions:
            for transition in state.transitions:
                target = html.escape(transition.target_state)
                trans_desc = html.escape(transition.description)
                priority = transition.priority

                xml_parts.append(f"    <transition>")
                xml_parts.append(f"      <targetState>{target}</targetState>")
                xml_parts.append(f"      <description>{trans_desc}</description>")
                xml_parts.append(f"      <priority>{priority}</priority>")

                # Add conditions if any
                if transition.conditions:
                    xml_parts.append(f"      <conditions>")
                    for condition in transition.conditions:
                        cond_desc = html.escape(condition.description)
                        xml_parts.append(f"        <condition>")
                        xml_parts.append(f"          <description>{cond_desc}</description>")
                        if condition.requires_context_keys:
                            xml_parts.append(f"          <requiredKeys>")
                            for key in condition.requires_context_keys:
                                xml_parts.append(f"            <key>{html.escape(key)}</key>")
                            xml_parts.append(f"          </requiredKeys>")
                        xml_parts.append(f"        </condition>")
                    xml_parts.append(f"      </conditions>")

                xml_parts.append(f"    </transition>")
        else:
            xml_parts.append("    <noTransitions>This state has no outgoing transitions.</noTransitions>")

        xml_parts.append("  </transitions>")

        # Add transition rules
        if available_states:
            xml_parts.append("  <rules>")
            xml_parts.append(
                f"    <rule>You MUST ONLY choose from the following valid target states: {available_states_str}</rule>")
            xml_parts.append(
                "    <rule>Do NOT invent or create new states that are not in the list of valid target states.</rule>")
            xml_parts.append(
                "    <rule>If you're unsure which state to transition to, stay in the current state.</rule>")
            xml_parts.append(
                f"    <rule>The current state is '{state_id}' - you can choose to stay here if needed.</rule>")
            xml_parts.append("  </rules>")

        # Add current context
        xml_parts.append("  <context>")
        if instance.context.data:
            for key, value in instance.context.data.items():
                safe_key = html.escape(str(key))
                safe_value = html.escape(str(value))
                xml_parts.append("    <item>")
                xml_parts.append(f"      <key>{safe_key}</key>")
                xml_parts.append(f"      <value>{safe_value}</value>")
                xml_parts.append("    </item>")
        else:
            xml_parts.append("    <empty>No context data available</empty>")
        xml_parts.append("  </context>")

        # Add conversation history with proper escaping
        recent_exchanges = instance.context.conversation.get_recent(5)
        if recent_exchanges:
            xml_parts.append("  <conversationHistory>")
            for exchange in recent_exchanges:
                for role, text in exchange.items():
                    sender = html.escape(role.lower())
                    message_text = html.escape(text)
                    xml_parts.append(f"    <message sender=\"{sender}\">{message_text}</message>")
            xml_parts.append("  </conversationHistory>")

        # Add example dialogue if available
        if state.example_dialogue:
            xml_parts.append("  <examples>")
            for exchange in state.example_dialogue:
                for role, text in exchange.items():
                    sender = html.escape(role.lower())
                    message_text = html.escape(text)
                    xml_parts.append(f"    <message sender=\"{sender}\">{message_text}</message>")
            xml_parts.append("  </examples>")

        # Define response schema elements outside the string interpolation
        schema_lines = [
            "      {",
            '        "transition": {',
            '          "target_state": "state_id",',
            '          "context_update": {"key1": "value1", "key2": "value2"}',
            '        },',
            '        "message": "Your message to the user",',
            '        "reasoning": "Your reasoning for this decision"',
            "      }"
        ]

        # Add response format instructions
        xml_parts.extend([
            "  <responseFormat>",
            "    <description>Respond with a JSON object with the following structure:</description>",
            "    <schema>"
        ])

        # Add schema lines
        for line in schema_lines:
            xml_parts.append(f"      {html.escape(line)}")

        xml_parts.extend([
            "    </schema>",
            "  </responseFormat>"
        ])

        # Add important guidelines
        xml_parts.extend([
            "  <guidelines>",
            "    <guideline>Collect all required information from the user's message</guideline>",
            "    <guideline>Only transition to a new state if all required information is collected or another transition is appropriate</guideline>",
            "    <guideline>Your message should be conversational and natural</guideline>",
            "    <guideline>Don't mention states, transitions, or context keys to the user</guideline>",
            f"    <guideline>Remember, you can ONLY choose from these valid target states: {available_states_str}</guideline>",
            "  </guidelines>",
            "</fsm>"
        ])

        prompt = "\n".join(xml_parts)
        logger.debug(f"System prompt length: {len(prompt)} characters")

        return prompt