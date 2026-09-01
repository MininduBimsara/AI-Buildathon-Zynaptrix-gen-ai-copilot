from unified_rag.config import settings
import sys

# Base URL for the backend API
API_URL = settings.api_url

def seed_machines():
    machines = [
        {
            "machine_id": "PUMP-001",
            "name": "Zynaptrix-9000 Turbo Pump",
            "location": "Hall A - Section 4",
            "manual_id": "Zynaptrix_9000"
        },
        {
            "machine_id": "LATHE-002",
            "name": "Precision Lathe X2",
            "location": "Hall B - Machine Shop",
            "manual_id": "lathe_manual_v3"
        },
        {
            "machine_id": "TURBINE-003",
            "name": "Gas Turbine G3",
            "location": "Power Block 1",
            "manual_id": "turbine_manual_v3"
        }
    ]

    print(f"🚀 Seeding machines to {API_URL}/api/machines ...")
    
    for machine in machines:
        try:
            # Use the /api prefix because routers are often prefixed or mounted
            # In our main_api.py, we include rag_router without prefix but tags exist.
            # Wait, let me check the mount in main_api.py.
            # router = APIRouter() in endpoints.py.
            # app.include_router(rag_router, tags=["Knowledge Base"]) in main_api.py.
            # So the endpoint is just /machines.
            
            resp = requests.post(f"{API_URL}/machines", json=machine)
            if resp.status_code in [200, 201]:
                print(f"✅ Created: {machine['name']} ({machine['machine_id']})")
            elif resp.status_code == 400 or "already exists" in resp.text.lower():
                print(f"ℹ️ Skipping: {machine['machine_id']} already exists.")
            else:
                print(f"❌ Failed: {machine['machine_id']} - Status {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    seed_machines()
