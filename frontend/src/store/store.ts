import { configureStore } from '@reduxjs/toolkit';
import copilotReducer from './slices/copilotSlice';
import machineReducer from './slices/machineSlice';
import simulatorReducer from './slices/simulatorSlice';
import ingestionReducer from './slices/ingestionSlice';

export const store = configureStore({
  reducer: {
    copilot: copilotReducer,
    machines: machineReducer,
    simulator: simulatorReducer,
    ingestion: ingestionReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
