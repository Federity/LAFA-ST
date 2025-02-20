# Workflow: Versioned Merger Commits with Stability Tracking
## 1. Initialization
Global Model: Initial parameters (e.g., commit_v0).

Stability Scores: Initialized to 1.0 for all parameters.

Pending Commit Buffer: Empty.

Active Commit Version: commit_v0 (tracked by the server).

## 2. Client Workflow (Asynchronous)
a. Client Download
Client requests the latest commit (commit_vN) from the server.

Receives:

current_model: Parameters from commit_vN.

current_stability: Stability scores from commit_vN.

current_version: Identifier vN.

## b. Local Training
Client trains on local data starting from commit_vN.

Computes:

Updated Parameters: client_params (trained on commit_vN).

Delta: client_delta = |client_params - commit_vN_params|.

## c. Client Upload
Client sends to the pending commit buffer:

client_params, client_delta, client_version (= vN).

## 3. Server Workflow (Batch Merging)
a. Batch Formation
Server periodically checks the pending buffer.

Groups updates by client_version:

Only merges updates based on the latest merged commit (e.g., vN).

Discards updates with client_version < current_version - staleness_threshold (e.g., 2).


b. Create Merger Commit
For valid updates (tagged with the latest vN):

Update Stability Scores:

Compute the batch-averaged delta:
Update stability scores.
Merge Parameters.

For each client in the batch:
Average updates:
Finalize Merger Commit:

Update current_version to v(N+1).

Broadcast commit_v(N+1) (parameters + updated stability scores).

Clear pending buffer of merged updates.

4. Key Properties
a. Version Linearity
All new client updates must derive from the latest merger commit (v(N+1)).

Example:

After merging v2, new clients download v3 and train only on v3.

b. Staleness Handling
Clients using outdated commits (e.g., v1 when v3 is active) are ignored if they exceed the staleness threshold.

c. Stability Inheritance
Stability scores evolve with each merger commit, preserving knowledge from prior merges.

d. Asynchrony
Clients train and submit updates independently, but merges are atomic (batched by version).

Example Timeline
Time T0:

Server initializes commit_v0.

Time T1:

Clients A, B train on commit_v0 and submit updates to the buffer.

Time T2:

Server merges updates into commit_v1.

Clients C, D now train on commit_v1.

Time T3:

Clients C, D submit updates to the buffer.

Client E (offline since T0) submits an update for commit_v0 → discarded (staleness=2).

Time T4:

Server merges updates into commit_v2.

Advantages
Consistency: All clients train on the latest merged commit.

Efficiency: Batched merging reduces server load.

Forgetting Prevention: Stability scores evolve with each merge, protecting critical parameters.

Scalability: Clients work asynchronously but align to linear versioning.