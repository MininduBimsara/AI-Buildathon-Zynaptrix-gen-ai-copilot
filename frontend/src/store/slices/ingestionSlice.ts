import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { addChatMessage } from './copilotSlice';

interface IngestionState {
  isUploading: boolean;
  uploadStatus: string | null;
  error: string | null;
}

const initialState: IngestionState = {
  isUploading: false,
  uploadStatus: null,
  error: null,
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const uploadManual = createAsyncThunk(
  'ingestion/uploadManual',
  async (payload: { manualId: string; file: File }, { dispatch }) => {
    const formData = new FormData();
    formData.append("manual_id", payload.manualId);
    formData.append("file", payload.file);

    const response = await fetch(`${API_BASE}/ingest-manual`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Upload failed');
    }

    // The upload call now only queues the job: a large manual makes one vision
    // call per figure and used to run for so long the browser gave up with
    // "Failed to fetch". Poll until the pipeline reports a terminal state.
    const POLL_MS = 3000;
    const MAX_MS = 60 * 60 * 1000; // an illustrated 150-page manual can take ~an hour
    const startedAt = Date.now();
    let unknownStreak = 0;

    while (Date.now() - startedAt < MAX_MS) {
        await new Promise((r) => setTimeout(r, POLL_MS));

        const statusRes = await fetch(
            `${API_BASE}/ingest-manual/status/${encodeURIComponent(payload.manualId)}`
        );
        if (!statusRes.ok) continue;
        const job = await statusRes.json();

        if (job.status === 'success') {
            dispatch(addChatMessage({
                role: 'agent',
                content: `📚 Manual "${payload.manualId}" successfully ingested and vectorized (${job.chunks ?? '?'} chunks).`
            }));
            return payload.manualId;
        }
        if (job.status === 'failed') {
            throw new Error(job.error || 'Ingestion failed');
        }
        // 'unknown' means the server restarted and lost the job registry.
        unknownStreak = job.status === 'unknown' ? unknownStreak + 1 : 0;
        if (unknownStreak >= 3) {
            throw new Error('Lost track of the ingestion job (backend restarted?)');
        }
    }

    throw new Error('Ingestion timed out. Check the backend logs — it may still be running.');
  }
);

const ingestionSlice = createSlice({
  name: 'ingestion',
  initialState,
  reducers: {
    clearUploadStatus(state) {
      state.uploadStatus = null;
      state.error = null;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(uploadManual.pending, (state) => {
        state.isUploading = true;
        state.uploadStatus = "Uploading to AI Pipeline...";
      })
      .addCase(uploadManual.fulfilled, (state) => {
        state.isUploading = false;
        state.uploadStatus = "Ingestion Successful!";
      })
      .addCase(uploadManual.rejected, (state, action) => {
        state.isUploading = false;
        state.error = action.error.message || 'Upload failed';
        state.uploadStatus = `Error: ${state.error}`;
      });
  },
});

export const { clearUploadStatus } = ingestionSlice.actions;
export default ingestionSlice.reducer;
