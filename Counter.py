import os
import json
from datetime import datetime
from fastapi import FastAPI, Path, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from google.cloud import firestore

# --- CONFIGURATION ---
# IMPORTANT: This assumes your environment is authenticated with Google Cloud
# (e.g., via 'gcloud auth application-default login' or by setting GOOGLE_APPLICATION_CREDENTIALS)
# The QR ID acts as a unique identifier for this specific robot/campaign.
DEFAULT_QR_ID = "robot_campaign_1" 

# The URL where the user will be redirected AFTER scanning the QR code
TARGET_REDIRECT_URL = "https://www.google.com/search?q=your+advertising+landing+page"

# Initialize FastAPI and Firestore
app = FastAPI(
    title="FastAPI QR Scan Tracker",
    description="Backend for tracking QR code scans in real-time using Firestore."
)
db = firestore.Client()
COLLECTION_NAME = "qr_scan_counters"

# --- HTML TEMPLATE FOR REAL-TIME DASHBOARD (Served by FastAPI) ---
# This uses client-side JS to listen to Firestore for real-time updates.
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QR Scan Dashboard - {qr_id}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js"></script>
    <script src="https://www.gstatic.com/firebasejs/11.6.1/firebase-firestore.js"></script>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f7f9fb;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
    </style>
</head>
<body>

    <div class="max-w-4xl w-full p-6 sm:p-10 bg-white rounded-xl shadow-2xl transition-all duration-300 text-center">
        <h1 class="text-4xl sm:text-5xl font-extrabold text-gray-900 mb-4">
            QR Scan Dashboard: {qr_id}
        </h1>
        <p class="text-lg text-gray-500 mb-8">
            Real-time count tracked by FastAPI.
        </p>
        
        <div class="bg-blue-600 p-8 rounded-xl shadow-xl transform hover:scale-[1.01] transition duration-300">
            <p class="text-xl font-medium text-blue-200 uppercase tracking-wider">Total Scans</p>
            <p id="scan-count" class="text-6xl sm:text-8xl font-black text-white mt-2 transition duration-500">0</p>
        </div>

        <div class="mt-10 p-6 bg-gray-50 rounded-lg text-left border border-gray-200">
            <h2 class="text-xl font-semibold text-gray-800 mb-3">Your QR Code Link</h2>
            <div class="mt-2 text-xs break-all bg-gray-100 p-3 rounded-md border border-dashed border-gray-300">
                <strong>Link for QR:</strong> <span id="qr-url-display">Loading...</span>
            </div>
            <p class="text-gray-600 text-xs mt-2">
                Scans automatically redirect to: <a href="{target_url}" target="_blank" class="text-blue-500 hover:underline">{target_url}</a>
            </p>
        </div>
        
        <p id="last-scan-time" class="text-xs text-gray-400 mt-4">Last recorded scan: N/A</p>
        <p id="error-message" class="text-red-500 mt-4 hidden">Connection Error.</p>
    </div>

    <script>
        // CLIENT-SIDE JAVASCRIPT FOR REAL-TIME DISPLAY
        // This relies on the firebaseConfig being passed from the environment (or hardcoded for simplicity here).
        // Since we are running in an immersive, we use the global variables provided.
        const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
        const firebaseConfig = typeof __firebase_config !== 'undefined' ? JSON.parse(__firebase_config) : null;

        if (!firebaseConfig) {{
             document.getElementById('error-message').textContent = "Firebase config missing. Cannot display real-time data.";
             document.getElementById('error-message').classList.remove('hidden');
        }} else {{
            const urlPathParts = window.location.pathname.split('/');
            const qrId = urlPathParts[urlPathParts.length - 1];

            // Set the QR URL for easy copying
            document.getElementById('qr-url-display').textContent = 
                `${{window.location.origin}}/scan/${{qrId}}`;

            const app = firebase.initializeApp(firebaseConfig);
            const dbClient = firebase.firestore();
            const docRef = dbClient.collection('{collection_name}').doc(qrId);

            docRef.onSnapshot((docSnap) => {{
                const countElement = document.getElementById('scan-count');
                const lastScanElement = document.getElementById('last-scan-time');

                if (docSnap.exists) {{
                    const data = docSnap.data();
                    const count = data?.count || 0;
                    const lastScan = data?.lastScan;

                    if (countElement) countElement.textContent = count.toLocaleString();

                    if (lastScanElement && lastScan) {{
                        const date = new Date(lastScan);
                        lastScanElement.textContent = `Last recorded scan: ${{date.toLocaleString()}}`;
                    }}
                }} else {{
                    if (countElement) countElement.textContent = '0';
                }}
                document.getElementById('error-message').classList.add('hidden');
            }}, (error) => {{
                console.error("Error setting up real-time listener:", error);
                document.getElementById('error-message').textContent = `Connection Error: ${{error.message}}`;
                document.getElementById('error-message').classList.remove('hidden');
            }});
        }}
    </script>
</body>
</html>
"""

# --- FASTAPI ROUTES ---

@app.get("/", include_in_schema=False)
async def root():
    """Redirects root to the default dashboard ID."""
    return RedirectResponse(url=f"/dashboard/{DEFAULT_QR_ID}")

@app.get("/dashboard/{qr_id}", response_class=HTMLResponse)
async def show_dashboard(
    qr_id: str = Path(..., title="The ID of the QR code campaign")
):
    """Serves the real-time HTML dashboard page."""
    
    # Fill the template placeholders with the actual QR ID and target URL
    html_content = DASHBOARD_HTML.format(
        qr_id=qr_id,
        target_url=TARGET_REDIRECT_URL,
        collection_name=COLLECTION_NAME
    )
    return HTMLResponse(content=html_content)


@app.get("/scan/{qr_id}", response_class=RedirectResponse)
async def scan_qr_code(
    qr_id: str = Path(..., title="The ID of the QR code campaign to increment")
):
    """
    Increments the scan count in Firestore using a transaction
    and redirects the user to the TARGET_REDIRECT_URL.
    """
    doc_ref = db.collection(COLLECTION_NAME).document(qr_id)

    def update_transaction(transaction, doc_ref):
        """Transaction function to safely increment the count."""
        try:
            doc_snapshot = doc_ref.get(transaction=transaction)
            
            # Initialize count if document does not exist
            current_count = doc_snapshot.get('count') if doc_snapshot.exists else 0
            new_count = current_count + 1

            # Update the document with the new count and timestamp
            transaction.set(doc_ref, {
                "count": new_count,
                "lastScan": datetime.now().isoformat(),
            })
            return new_count
        except Exception as e:
            # Important to log the exception for debugging
            print(f"Firestore Transaction failed for {qr_id}: {e}")
            raise e # Re-raise to trigger transaction retry/failure

    try:
        # Run the transaction
        new_count = db.transaction(update_transaction, max_attempts=5)(doc_ref)
        print(f"Scan recorded for '{qr_id}'. New count: {new_count}")
    except Exception as e:
        # Log and handle transaction failure, but still redirect the user
        print(f"Failed to record scan for '{qr_id}' after retries. Error: {e}")

    # Always redirect the user, even if the count failed (user experience priority)
    return RedirectResponse(url=TARGET_REDIRECT_URL, status_code=302)


# --- RUNNER INSTRUCTIONS ---
# To run this file:
# 1. Install dependencies: pip install "fastapi[all]" google-cloud-firestore
# 2. Ensure your Google Cloud credentials are set up (e.g., 'gcloud auth application-default login')
# 3. Run the application: uvicorn main:app --reload
