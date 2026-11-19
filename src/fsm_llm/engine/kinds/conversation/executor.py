"""``ConversationExecutor`` and ``ConversationPass1``: the conversation kind's Pass 1.

Layer ``engine`` (plan d454fd62, step 80, D-030). ``ConversationExecutor`` is what the registry
holds for the ``conversation`` kind (and, as the passive fallback, for ``synthesize``). It owns the
conversation Pass-1 machinery: the field, bulk and classification extraction and the
post-transition extraction, which were methods of ``MessagePipeline`` until step 80.

The executor is a stateless singleton shared by every pipeline of the process (``executor_for``
returns the same object), so it cannot hold the pipeline. Each call builds a ``ConversationPass1``
bound to the services of the calling pipeline (``_PipelineServices``, the ``pipeline`` argument)
and calls the same method bodies the pipeline used to carry. ``ConversationPass1`` holds nothing
else, lives for one call and is not shared between threads.

Interface contract (``ConversationPass1``): ``ConversationPass1(services)``; the private methods
keep the names and signatures they had on the pipeline (the unit tests call them through this
class); ``services`` is the pipeline (or a test double with the members of ``_PipelineServices``).
"""

from __future__ import annotations

from fsm_llm.core.definitions import DataExtractionResponse, FSMInstance, State

from ..base import StateExecutor
from ..services import _PipelineServices
from .bulk import ConversationBulkMixin
from .classification import ConversationClassificationMixin
from .extraction import ConversationExtractionMixin
from .fields import ConversationFieldsMixin
from .post_transition import ConversationPostTransitionMixin


class ConversationPass1(
    ConversationPostTransitionMixin,
    ConversationExtractionMixin,
    ConversationBulkMixin,
    ConversationFieldsMixin,
    ConversationClassificationMixin,
):
    """The conversation kind's Pass-1 methods, bound to one pipeline's services."""

    def __init__(self, services: _PipelineServices) -> None:
        self._services = services


class ConversationExecutor(StateExecutor):
    """The conversation kind: field, bulk and classification extraction.

    Owns the machinery (``ConversationPass1``, built per call over the pipeline's services,
    plan d454fd62 step 80, D-030); also the passive executor of ``synthesize``.
    """

    # DECISION plan-2026-11-12T073150-d454fd62/D-030
    # The conversation Pass 1 is owned here and bound to ONE pipeline's services per call
    # (``ConversationPass1(pipeline)``). Do NOT keep the pipeline on this executor or cache the
    # machinery between calls: the executor is a singleton shared by every pipeline of the
    # process and by every thread, so a stored pipeline would run another API's turn on the
    # first one's ``llm_interface``. Do NOT import ``engine/turn`` (not even under
    # ``TYPE_CHECKING``, which the import contract does not see) to type ``pipeline``, and do
    # NOT move the methods back onto ``MessagePipeline`` as mixins: the kinds sit below the
    # turn and ``_PipelineServices`` is the whole contract. See D-030.
    def extract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        response: DataExtractionResponse = ConversationPass1(
            pipeline
        )._execute_data_extraction(instance, user_message, conversation_id)
        return response

    def after_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        ConversationPass1(pipeline)._extract_after_transition(
            instance,
            user_message,
            conversation_id,
            extraction_response,
            previous_state,
            new_state,
        )

    # DECISION plan-2026-11-19T193704-f150ec28/D-012
    # ``extract`` and ``after_transition`` above stay the SYNC route and keep their direct reads
    # of ``_execute_data_extraction`` and ``_extract_after_transition``; these two twins are the
    # async route and await the async bodies. Do NOT fold ``extract`` into
    # ``run_sync(self.aextract(...))``: in sync mode the pipeline reaches the executor through
    # ``acall(executor, "extract", ...)``, so a patch of ``extract`` must stay the thing that is
    # called (``test_regression_transition_eval``), and ``test_executors`` pins which names this
    # module reads.
    async def aextract(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        state: State,
    ) -> DataExtractionResponse:
        return await ConversationPass1(pipeline)._aexecute_data_extraction(
            instance, user_message, conversation_id
        )

    async def aafter_transition(
        self,
        pipeline: _PipelineServices,
        instance: FSMInstance,
        user_message: str | None,
        conversation_id: str,
        extraction_response: DataExtractionResponse,
        previous_state: str | None,
        new_state: State,
    ) -> None:
        await ConversationPass1(pipeline)._aextract_after_transition(
            instance,
            user_message,
            conversation_id,
            extraction_response,
            previous_state,
            new_state,
        )
