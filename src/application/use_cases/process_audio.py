"""Use case: turn an audio file into a diarized Transcript."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.application.ports.diarizer_port import DiarizerPort
from src.application.ports.transcriber_port import TranscriberPort
from src.domain.entities.meeting import Meeting, MeetingStatus
from src.domain.entities.transcript import Transcript
from src.domain.repositories.meeting_repository import MeetingRepository
from src.domain.repositories.transcript_repository import TranscriptRepository
from src.domain.services.transcript_merger import TranscriptMerger
from src.domain.value_objects.audio_file import AudioFile


@dataclass(frozen=True, slots=True)
class ProcessAudioInput:
    meeting_id: UUID


@dataclass(frozen=True, slots=True)
class ProcessAudioOutput:
    transcript_id: UUID


class ProcessAudioUseCase:
    """Pipeline orchestrator: ASR + diarization + merge.

    Pure orchestration — no model knowledge here. Each external dependency is
    injected via its port.
    """

    def __init__(
        self,
        transcriber: TranscriberPort,
        diarizer: DiarizerPort,
        merger: TranscriptMerger,
        meetings: MeetingRepository,
        transcripts: TranscriptRepository,
    ) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._merger = merger
        self._meetings = meetings
        self._transcripts = transcripts

    def execute(self, payload: ProcessAudioInput) -> ProcessAudioOutput:
        meeting = self._load_meeting(payload.meeting_id)
        audio = AudioFile(path=meeting.audio_path)

        try:
            transcript = self._run_pipeline(meeting, audio)
        except Exception as exc:
            meeting.mark_failed(f"audio processing failed: {exc}")
            self._meetings.update(meeting)
            raise

        return ProcessAudioOutput(transcript_id=transcript.id)

    def _run_pipeline(self, meeting: Meeting, audio: AudioFile) -> Transcript:
        meeting.transition_to(MeetingStatus.TRANSCRIBING)
        self._meetings.update(meeting)
        raw_segments = self._transcriber.transcribe(audio)

        meeting.transition_to(MeetingStatus.DIARIZING)
        self._meetings.update(meeting)
        speaker_turns = self._diarizer.diarize(audio)

        transcript = self._merger.merge(
            meeting_id=meeting.id,
            raw_segments=raw_segments,
            speaker_turns=speaker_turns,
        )
        self._transcripts.add(transcript)
        meeting.attach_transcript(transcript.id)
        self._meetings.update(meeting)
        return transcript

    def _load_meeting(self, meeting_id: UUID) -> Meeting:
        meeting = self._meetings.get(meeting_id)
        if meeting is None:
            raise LookupError(f"Meeting {meeting_id} not found")
        return meeting
