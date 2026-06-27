# ☁️ Microsoft Azure Student Plan — Deployment Guide

Deploy the XAI Dashboard to an Azure **Standard_B4s_v2 VM** (4 vCPUs, 16GB RAM, 6400 IOPS, x86_64) using your Azure Student Plan credits for a cloud vs. local performance comparison.

---

## ⚠️ Important: Azure Credits Management
Provisioning a `Standard_B4s_v2` VM costs approximately **$140.16/month**. Under the Azure Student Plan, this is deducted from your **$100 free credit**.
- **To avoid running out of credits:** Proactively **Deallocate (Stop)** the VM in the Azure Portal when you are not actively running tests or gathering thesis data. 
- Stopping the VM stops compute charges (you will only pay a few cents/month for the OS disk storage).

---

## Prerequisites

- An active Microsoft Azure Student account (includes $100 in credits)
- An SSH key pair (or generated during VM creation)
- Git installed locally

---

## Step 1: Create the Azure VM

1. Log in to the [Azure Portal](https://portal.azure.com/).
2. Search for **Virtual machines** and click **Create** → **Azure virtual machine**.
3. Configure the VM under **Basics**:
   - **Subscription:** `Azure for Students`
   - **Resource Group:** Click *Create new* → name it `xai-rg`
   - **Virtual machine name:** `xai-dashboard`
   - **Region:** Select a region near you, according to the [Azure Student Plan's policy](https://portal.azure.com/#view/Microsoft_Azure_Policy/PolicyMenuBlade.MenuView/~/Assignments). Click **Allowed resource deployment regions**. The **Parameter value** entry down below contains a list of allowed regions. (In my case, those are: **\["swedencentral","polandcentral","uaenorth","italynorth","francecentral"\]**).
   - **Image:** `Ubuntu Server 22.04 LTS - x64 Gen2` (or `24.04 LTS`)
   - **Size:** Click *See all sizes* and select **B4s_v2** from the **B-Series v2** tab (4 vCPUs, 16 GB RAM, 6400 IOPS).
   - **Authentication type:** `SSH public key`
   - **Username:** `azureuser`
   - **Key pair name:** `xai-key`
4. Under **Inbound port rules**:
   - **Public inbound ports:** `Allow selected ports`
   - **Select inbound ports:** `SSH (22)`
5. Click **Review + create**, then click **Create** (make sure to download the generated `.pem` private key file).

### Configure Inbound Rules (Network Security Group)

By default, Azure blocks all external ports except SSH (22). You must allow port 8501 (Streamlit) and port 11434 (Ollama):

1. Go to your newly created Virtual Machine page.
2. Under the left menu, select **Networking** (or **Network settings**).
3. Click **Add inbound port rule** and configure:
   - **Source:** `Any`
   - **Source port ranges:** `*`
   - **Destination:** `Any`
   - **Destination port ranges:** `8501`
   - **Protocol:** `TCP`
   - **Action:** `Allow`
   - **Priority:** `310`
   - **Name:** `Allow-Streamlit`
4. Click **Add**. 
5. Repeat for port `11434` (Ollama API - optional) if you wish to run benchmarks directly from your local machine to the cloud. Leave priority at default (will be higher, like 320).

---

## Step 2: Provision the VM

Run the setup script from your **local machine** using PowerShell or bash:

```bash
ssh -i <path-to-xai-key.pem> azureuser@<vm-public-ip> 'bash -s' < deploy/setup-cloud.sh
```

This updates Ubuntu package databases, installs Docker + Docker Compose, configures firewall access, and clones the repository.

---

## Step 3: First Deploy (Manual)

SSH into your Azure VM and start the dockerized dashboard:

```bash
ssh -i <path-to-xai-key.pem> azureuser@<vm-public-ip>
cd ~/xai-app

# Fill in Kaggle credentials if you want to download datasets dynamically
nano .env

# Build and start (first time takes ~10-15 minutes to pull base images and models)
docker compose -f docker/docker-compose.cloud.yml up -d --build --remove-orphans

# Watch container initialization logs
docker compose -f docker/docker-compose.cloud.yml logs -f
```

Once the logs show `All models pulled. Ollama is ready!`, the app is live at:
```
http://<vm-public-ip>:8501
```

---

## Step 4: Connect Jenkins for Automatic Deploys

### Add Credentials to Jenkins

1. Go to **Jenkins** → **Manage Jenkins** → **Credentials** → **System** → **Global**
2. Add two **Secret text** credentials:
   - **ID:** `azure-cloud-vm-ip` — **Value:** Your Azure VM's public IP
   - **ID:** `azure-cloud-ssh-key` — **Value:** The absolute path to the private SSH key inside the Jenkins host container (e.g., `/var/jenkins_home/.ssh/azure_key`)

3. Copy your SSH private key file into the Jenkins container filesystem:
   ```bash
   docker cp xai-key.pem xai-jenkins:/var/jenkins_home/.ssh/azure_key
   docker exec xai-jenkins chmod 600 /var/jenkins_home/.ssh/azure_key
   ```

Now, every commit pushed to the `main` branch will trigger: **Lint → Test → Build → Deploy (Local) → Deploy (Cloud Azure) → Smoke Test**.

---

## Performance Comparison (Thesis Research)

Use this table to benchmark local CPU/GPU vs. Cloud CPU (Azure Standard_B4s_v2) inference performance:

| Metric | Local (GPU/CPU) | Cloud (Azure B4s_v2 CPU) |
|---|---|---|
| LLM tokens/sec (llama3.1-8B-Q4) | ___ | ___ |
| VLM transcription time (qwen2.5vl) | ___ s | ___ s |
| CNN training time (EMNIST, 5 epochs) | ___ s | ___ s |
| App cold start (docker compose up) | ___ s | ___ s |
| Streamlit page load (first paint) | ___ ms | ___ ms |

### How to Measure LLM tokens/sec

Run this command inside the VM or locally (replacing `localhost` with your VM IP):

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b-instruct-q4_K_M",
  "prompt": "Explain explainable AI in 200 words.",
  "stream": false
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]/d[\"eval_duration\"]*1e9:.1f} tokens/sec')"
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `docker compose` not found | Run `sudo apt update && sudo apt install docker-compose-plugin` |
| Port 8501 not accessible | Check the Azure Inbound Port Security rules for `8501` |
| Ollama OOM (Out Of Memory) | B4s_v2 has 16GB RAM. If running multiple containers, set up a swap file: <br>`sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |

---

## Cost Summary

| Resource | Hourly Price | Estimated Monthly Cost (24/7) |
|---|---|---|
| VM (Standard_B4s_v2 - 4 vCPUs, 16GB) | ~$0.08 | ~$60.00 (Fully covered by Student credits) |
| Premium SSD OS Disk (64 GB) | ~$0.007 | ~$5.00 (Fully covered by Student credits) |
| **Total** | | **~$65/month** (Charged to credits) |
