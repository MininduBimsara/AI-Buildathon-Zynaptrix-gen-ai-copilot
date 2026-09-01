import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface SimulatorState {
  activeSimulators: string[];
  loading: boolean;
  error: string | null;
}

const initialState: SimulatorState = {
  activeSimulators: [],
  loading: false,
  error: null,
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const fetchSimulatorStatus = createAsyncThunk('simulator/fetchStatus', async () => {
    const response = await fetch(`${API_BASE}/api/simulator/status`);
    if (!response.ok) throw new Error('Failed to fetch status');
    const data = await response.json();
    return data.active_simulators as string[];
});

// A null/undefined selection stringifies to the literal "null" inside a template
// literal, which the backend then treats as a real machine id.
const assertMachineId = (machineId: string) => {
    const id = (machineId ?? '').trim();
    if (!id || id === 'null' || id === 'undefined') {
        throw new Error('Select a machine before starting the simulator.');
    }
    return encodeURIComponent(id);
};

export const startSimulator = createAsyncThunk('simulator/start', async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/simulator/start?machine_id=${assertMachineId(machineId)}`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to start simulator');
    const data = await response.json();
    return data;
});

export const stopSimulator = createAsyncThunk('simulator/stop', async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/simulator/stop?machine_id=${machineId}`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to stop simulator');
    const data = await response.json();
    return data;
});

export const injectAnomaly = createAsyncThunk('simulator/inject', async (payload: { machineId: string, anomalyType: string }) => {
    const response = await fetch(`${API_BASE}/api/simulator/inject?machine_id=${payload.machineId}&anomaly_type=${payload.anomalyType}`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to inject anomaly');
    const data = await response.json();
    return data;
});

const simulatorSlice = createSlice({
  name: 'simulator',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchSimulatorStatus.fulfilled, (state, action) => {
        state.activeSimulators = action.payload;
      })
      .addMatcher(
        (action) => action.type.endsWith('/fulfilled') && action.type.startsWith('simulator/'),
        (state) => {
            // Re-fetch status after start/stop to be safe
            // Or we could optimistically update
        }
      );
  },
});

export default simulatorSlice.reducer;
