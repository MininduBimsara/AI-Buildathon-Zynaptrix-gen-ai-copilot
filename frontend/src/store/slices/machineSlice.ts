import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

export interface Machine {
  machine_id: string;
  name: string;
  location: string;
  manual_id: string;
}

export interface SensorMeta {
  sensor_id: string;
  sensor_name: string;
  icon_type: string;
  unit: string;
}

interface MachineState {
  machines: Machine[];
  machineConfigs: Record<string, SensorMeta[]>;
  currentMachineId: string;
  loading: boolean;
  error: string | null;
}

const initialState: MachineState = {
  machines: [],
  machineConfigs: {},
  currentMachineId: 'PUMP-001',
  loading: false,
  error: null,
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const fetchMachines = createAsyncThunk('machines/fetchMachines', async () => {
    const response = await fetch(`${API_BASE}/api/machines`);
    if (!response.ok) throw new Error('Failed to fetch machines');
    return (await response.json()) as Machine[];
});

export const registerMachine = createAsyncThunk('machines/registerMachine', async (formData: FormData) => {
    const response = await fetch(`${API_BASE}/api/machines`, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) throw new Error('Failed to register machine');
    return (await response.json()) as Machine;
});

export const deleteMachine = createAsyncThunk('machines/deleteMachine', async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/machines/delete/${machineId}`, {
        method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to decommission machine');
    return machineId;
});

export const fetchMachineConfig = createAsyncThunk('machines/fetchConfig', async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/simulator/machine/${machineId}/config`);
    if (!response.ok) throw new Error('Failed to fetch machine config');
    const data = await response.json();
    // API returns { sensors: [...], sensors_meta: [{sensor_id, sensor_name, icon_type, unit}] }
    const sensorsMeta: SensorMeta[] = data.sensors_meta || data.sensors?.map((id: string) => ({
        sensor_id: id, sensor_name: id, icon_type: 'generic', unit: 'units'
    })) || [];
    return { machineId, sensorsMeta };
});

const machineSlice = createSlice({
  name: 'machines',
  initialState,
  reducers: {
    setCurrentMachineId(state, action: PayloadAction<string>) {
      state.currentMachineId = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMachines.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchMachines.fulfilled, (state, action) => {
        state.loading = false;
        state.machines = action.payload;
      })
      .addCase(fetchMachines.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Unknown error';
      })
      .addCase(registerMachine.pending, (state) => {
        state.loading = true;
      })
      .addCase(registerMachine.fulfilled, (state, action) => {
        state.loading = false;
        // Check if machine already exists in local list, update it, otherwise add it
        const index = state.machines.findIndex(m => m.machine_id === action.payload.machine_id);
        if (index !== -1) {
            state.machines[index] = action.payload;
        } else {
            state.machines.push(action.payload);
        }
      })
      .addCase(registerMachine.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Registration failed';
      })
      .addCase(deleteMachine.pending, (state) => {
        state.loading = true;
      })
      .addCase(deleteMachine.fulfilled, (state, action) => {
        state.loading = false;
        state.machines = state.machines.filter(m => m.machine_id !== action.payload);
        // If current machine was deleted, reset it to the first available or default
        if (state.currentMachineId === action.payload) {
            state.currentMachineId = state.machines.length > 0 ? state.machines[0].machine_id : 'PUMP-001';
        }
      })
      .addCase(deleteMachine.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Deletion failed';
      })
      .addCase(fetchMachineConfig.fulfilled, (state, action) => {
        // Stores the SensorMeta[] for the machine
        state.machineConfigs[action.payload.machineId] = action.payload.sensorsMeta;
      });
  },
});

export const { setCurrentMachineId } = machineSlice.actions;
export default machineSlice.reducer;
