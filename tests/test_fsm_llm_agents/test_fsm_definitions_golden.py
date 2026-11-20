"""Golden hashes of every FSM builder in ``agents/fsm_definitions/``.

Each case calls a builder with one representative argument set and pins the
canonical JSON (``json.dumps(d, sort_keys=True)``) sha256 and, for a full FSM,
the ``fsm_id`` core would mint (``API.from_definition`` hashes
``FSMDefinition.model_dump(exclude_defaults=True)``). Captured at 7cb4c86,
before the dict shapes were collapsed into helpers (plan 5980fbb7 step 13).

``conditions: []`` and an absent ``conditions`` load as different transitions
and give different ``fsm_id``s, so a helper that normalises one into the other
fails here.

``_approval_think_transition`` was inlined into ``_think_transitions`` (one
caller); the ``*_approval`` cases cover its edge.

The case table itself lives in ``tests/executables_corpus.py`` (plan ace7a8c8,
F1.2), beside the readable text fixtures built from it.

Do NOT regenerate a value to make a failure go away: a difference means a
builder now emits a different prompt or routing contract.

Re-pinned once for F-AR-05 (plan 0068b055 step 7, D-013): ``adapt``,
``adapt_depth``, ``orchestrator`` and ``orchestrator_keys`` moved because the
six intermediate states (orchestrate, delegate, collect; attempt, assess,
decompose) lost their ``response_instructions``. The dicts were diffed before
the re-pin: only those six fields changed, nothing else.
"""

from __future__ import annotations

import pytest

from fsm_llm.core.definitions import FSMDefinition, fsm_id_of
from tests.executables_corpus import CASES, canonical_sha256

#: case name -> (sha256 of the canonical JSON, fsm_id or "" for a fragment)
_GOLDEN: dict[str, tuple[str, str]] = {
    "adapt": (
        "63b05f8048f8238f053d66f4ee93e1301f7fab8990627a9fa7ab18c32b1cb61f",
        "fsm_adapt_agent_df1964dd",
    ),
    "adapt_depth": (
        "c08b304e2723ebdf2b0e76b13e542000a9b21bd3f38d014adbc7c79495fa3166",
        "fsm_adapt_agent_2e615993",
    ),
    "await_approval_state": (
        "393e8623fb20ed25ddea4f976ed2feae6905132dc4606d18ad0ec26c485bd08a",
        "",
    ),
    "conclude_logic": (
        "0dbd0c38d9d910979b5f2eb9b3ea6370e0969b8d1eabd43b6cba0618f3b1e529",
        "",
    ),
    "conclude_logic_flag": (
        "9f44111b718849d4ed185fdab4d43b81907baa958b5883fef9703e763388dae3",
        "",
    ),
    "debate": (
        "87b847be85714e49f9a7f5051f5a43b845b787eca0c0dbaef91c221f74147de1",
        "fsm_debate_agent_5a2b96c0",
    ),
    "debate_personas": (
        "101d8dd3b7607778a3d9bc772cfe14bb08a910e12aac411ffd6ab80129ac87c6",
        "fsm_debate_agent_3edd7bc0",
    ),
    "evalopt": (
        "2df5b1e0f1f267f86064c7cedaf8eb27c2fc87e5e02e48c976dabd11a8086ebd",
        "fsm_evalopt_agent_92555c30",
    ),
    "evalopt_task": (
        "d3fd79b1d4bb62e4de980c31148602b5f1b5c9d8eda7c25b959b724bb9822d4a",
        "fsm_evalopt_agent_09b53f1e",
    ),
    "finalize": (
        "6f3a0eb19050a46126fcdc6ac6ec9959343609d92746a2fb26ada398b839e4d7",
        "",
    ),
    "finalize_default_description": (
        "af241c6abe316938a9f78e6908c0df0c343d93529f1c5adc53a2f7e821ef2c0e",
        "",
    ),
    "flag_condition": (
        "4c3e2fe7e763cfb8c6d187f6c717fad62551a50c4c7fd59e20816ab24641ad1b",
        "",
    ),
    "loop_field_context_keys": (
        "479befedfae79cf900993283f1fd2bc6041ceec6342de05c6413b0dcb360378c",
        "",
    ),
    "loop_field_context_keys_extra": (
        "3ab03b0c58a5c064ecedf093cfcd13f12f3c2f8af72cef3de498547a025bcecf",
        "",
    ),
    "maker_checker": (
        "428c6017c12b922d9e866077399a91e9e78eaf12f2696f03ad1237c7b4846656",
        "fsm_maker_checker_agent_1f77ae6c",
    ),
    "maker_checker_cap": (
        "5568692033b078507f0ee2f24ca6f15ebff606b541d299cf58f17257af6cef9c",
        "fsm_maker_checker_agent_e6a19d69",
    ),
    "meta_builder": (
        "3b886ceafae99c9a9760390b86cdf07661712c1458a2f2765523755774732c1c",
        "fsm_meta_builder_bb853333",
    ),
    "native_fc": (
        "71daf584a917f3098d0034ac562e743ee58835992510c67f89f731b5c7f53c2e",
        "fsm_native_fc_c3dde2da",
    ),
    "native_fc_forced": (
        "838169492505f6bee4e07d451cae99c7e313945a116689548ebb0e4384afa8d7",
        "fsm_native_fc_a125a7ed",
    ),
    "native_fc_forced_repair": (
        "e1e0fce7b118035984468defed36ebc96bb010b9cf6f771ff8b7280e58c64e15",
        "fsm_native_fc_e477e486",
    ),
    "native_fc_repair": (
        "6be70516e6ae1a95d1885991f4e6bd74933c38651614de0f4d10cddea1643e82",
        "fsm_native_fc_fd75aff1",
    ),
    "native_fc_undeclared_forced": (
        "71daf584a917f3098d0034ac562e743ee58835992510c67f89f731b5c7f53c2e",
        "fsm_native_fc_c3dde2da",
    ),
    "orchestrator": (
        "cd5d125a4c5a28980cd30a1b04d75fcdace1365b1d06cd95e1d8030b6e78a24d",
        "fsm_orchestrator_agent_4631f149",
    ),
    "orchestrator_keys": (
        "637823f3f2ec86532eac83f488ea4c0e17857937ed056fe12004b97011798af2",
        "fsm_orchestrator_agent_a20a809d",
    ),
    "parallel_react": (
        "73753f1d2acac8a0ed36d6328f66310132c9751496acd959f873f7f1d1c64b48",
        "fsm_ParallelReactAgent_2fafa029",
    ),
    "parallel_react_non_model_schema": (
        "9756f915555f015e2fe9281f7a806818eb1c315ee74868e32430f8ea24c01ebd",
        "fsm_ParallelReactAgent_05904a2d",
    ),
    "parallel_react_schema_keys": (
        "2d3d2c76dad9737d7f6c7cb1392d850c5fe7067a7b6b8eab869e726b244ca3e4",
        "fsm_ParallelReactAgent_2d815551",
    ),
    "plan_execute": (
        "d673c432787899300d40da441e7a85e34b9812edf0bfdf7f5484cf643efec0a6",
        "fsm_plan_execute_agent_10d97c73",
    ),
    "plan_execute_empty": (
        "d673c432787899300d40da441e7a85e34b9812edf0bfdf7f5484cf643efec0a6",
        "fsm_plan_execute_agent_10d97c73",
    ),
    "plan_execute_tools": (
        "f605734e0104e30055056563edf1a46ce4214140a043608e10af5713378a332c",
        "fsm_plan_execute_agent_e51b5075",
    ),
    "prompt_chain_gates": (
        "d50743e9f63e349e09a52c3803817d83a8c6fd1a1b91287b6fd7d6f6c0b4006c",
        "fsm_prompt_chain_agent_1fa788e1",
    ),
    "prompt_chain_one": (
        "02873e8ca73a495566ed982bf6958597ba3728c855e86c5b53d0e0fb7c43f74d",
        "fsm_prompt_chain_agent_7c51b103",
    ),
    "react": (
        "3fd2a6492045561d8e8628bd3bcc927608528643b1e79b8b62644e15ed84a924",
        "fsm_react_agent_5cc3ffd3",
    ),
    "react_approval": (
        "bb16de22ad89d76c605c4fb8fe7cca4d59c92d57f453a14afe525ec9f08ed108",
        "fsm_react_agent_9e97feb8",
    ),
    "react_classification": (
        "c7472bf36b2510d98ac31550726f009b34cbd69e21eeac9298690433b5031dd8",
        "fsm_react_agent_2beb738c",
    ),
    "react_classification_approval": (
        "e9460a24905ecd59bbaaeebb7556419af43f810106fec45d2c38b5f672ea3abe",
        "fsm_react_agent_b9f6e61e",
    ),
    "react_non_model_schema": (
        "ff0223e3a877fe5178903058e6a3324c879d36e3c8e4b444c7d64f3ca777d7e9",
        "fsm_react_agent_24d3f1e5",
    ),
    "react_schema": (
        "3575226c9c0ce24e0c7482bdca24ebb4f2f95946f55960424b24cf7391ffe3d5",
        "fsm_react_agent_4b00b247",
    ),
    "react_task_keys": (
        "72d5c03b5b5743a180b3f862405eb175dfac0cebdfbb473a03046d823cbc135f",
        "fsm_react_agent_ef3b14a6",
    ),
    "reflexion": (
        "16994cd58eddae9299df3c247eac52c786f57853b11af1fccbf8d9d5f50a8da5",
        "fsm_reflexion_agent_eb91dbc3",
    ),
    "reflexion_approval": (
        "9a633528aafa6a35221220d1240d80d946ca42bc12ba936437cf861d6a2b3986",
        "fsm_reflexion_agent_7efb3140",
    ),
    "reflexion_empty": (
        "354a806cfba6b1a5d1659f771e8e9b85ab63e404c84bcdfdbdc73ae91ec4db33",
        "fsm_reflexion_agent_f96d7a2d",
    ),
    "reply_kind_condition": (
        "6d86bb241f242498f02ef25398d0c3dc5fa3d1e946854307e660c2e379553a73",
        "",
    ),
    "rewoo": (
        "561bcd368a96bd22a1bcd8038e2cd62d4d22c0cf6abe4b50489557be5ae52167",
        "fsm_rewoo_agent_8b17ae8f",
    ),
    "rewoo_task_keys": (
        "e3a088f4914c50d08c487fe97584d76ef872385faf008ddba2393918ec343435",
        "fsm_rewoo_agent_0099bc64",
    ),
    "self_consistency": (
        "f0f9e85e97a224be48ee11161aadb870b5a43e94088f885b8587ebefd65b8cd5",
        "fsm_self_consistency_sample_b1456cc8",
    ),
    "self_consistency_task": (
        "dcf2033047804af4f5ffa886c7633b36760224896667f85d4f9a30cac4df956f",
        "fsm_self_consistency_sample_d1c56915",
    ),
    "think_fields": (
        "63a1b85d464f03fdd089906ba900f8ecffea824a91858973fdf7cee3b651a98d",
        "",
    ),
    "think_fields_no_name": (
        "d54e2dbaa4ae50822270e6626eb9f3f0d3a3d03308041d71ab70ef7a052b70ec",
        "",
    ),
    "tool_selection": (
        "5784ab0b6a088748fae20b2c7ad23fac0cdaced80fc9cb6d4be0b5b71e1d0050",
        "",
    ),
    "tool_selection_no_name": (
        "3ab71852345a3ef8844d9212965d9da8fb80981ed8e02233ce0c53dd8bd912ba",
        "",
    ),
}


def _measure(name: str) -> tuple[str, str]:
    build, is_fsm = CASES[name]
    value = build()
    return canonical_sha256(value), fsm_id_of(FSMDefinition(**value)) if is_fsm else ""


def test_every_case_has_a_golden_value():
    assert set(_GOLDEN) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_builder_output_is_unchanged(name):
    assert _measure(name) == _GOLDEN[name]
