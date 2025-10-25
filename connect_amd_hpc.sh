#!/bin/bash
# AMD HPC Fund Research Cloud - Connection Helper Script
# User: pedramnj

echo "🚀 AMD HPC Fund Research Cloud Connection Helper"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HPC_USER="pedramnj"
HPC_HOST="hpcfund.amd.com"
SSH_KEY="$HOME/.ssh/id_ed25519"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    echo "Please ensure your SSH key is properly set up."
    exit 1
fi

# Check key permissions
KEY_PERMS=$(stat -f %A "$SSH_KEY" 2>/dev/null || stat -c %a "$SSH_KEY" 2>/dev/null)
if [ "$KEY_PERMS" != "600" ]; then
    echo -e "${YELLOW}⚠️  Fixing SSH key permissions...${NC}"
    chmod 600 "$SSH_KEY"
fi

echo -e "${GREEN}✓${NC} SSH key found: $SSH_KEY"
echo ""

# Display public key fingerprint
echo "Your SSH key fingerprint:"
ssh-keygen -lf "$SSH_KEY" 2>/dev/null
echo ""

# Ask what user wants to do
echo "What would you like to do?"
echo "1) Test connection"
echo "2) Connect to login node"
echo "3) Connect with tmux session"
echo "4) Connect with screen session"
echo "5) Show SLURM quick commands"
echo "6) Update SSH key with AMD (show instructions)"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Testing connection...${NC}"
        ssh -o ConnectTimeout=10 -o BatchMode=yes ${HPC_USER}@${HPC_HOST} "echo 'Connection successful!'; hostname; whoami; date"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Connection successful!${NC}"
        else
            echo -e "${RED}❌ Connection failed!${NC}"
            echo ""
            echo "Possible reasons:"
            echo "1. SSH key not yet activated by AMD administrators"
            echo "2. SSH key mismatch (see AMD_HPC_ACCESS.md)"
            echo "3. Network/firewall issue"
            echo ""
            echo "To debug, run:"
            echo "  ssh -v ${HPC_USER}@${HPC_HOST}"
        fi
        ;;
    2)
        echo -e "\n${GREEN}Connecting to AMD HPC login node...${NC}"
        echo "Remember: DO NOT run intensive jobs on login node!"
        echo ""
        ssh ${HPC_USER}@${HPC_HOST}
        ;;
    3)
        echo -e "\n${GREEN}Connecting with tmux session...${NC}"
        ssh -t ${HPC_USER}@${HPC_HOST} "tmux new-session -A -s main"
        ;;
    4)
        echo -e "\n${GREEN}Connecting with screen session...${NC}"
        ssh -t ${HPC_USER}@${HPC_HOST} "screen -R main"
        ;;
    5)
        echo -e "\n${GREEN}SLURM Quick Commands:${NC}"
        echo "======================================"
        echo ""
        echo "📊 Queue & Job Info:"
        echo "  squeue                  # View all jobs"
        echo "  squeue -u pedramnj     # View your jobs"
        echo "  sinfo                  # View partitions"
        echo "  sacct                  # Job accounting"
        echo ""
        echo "🚀 Submit Jobs:"
        echo "  sbatch job.sh          # Submit batch job"
        echo "  salloc -N 1 -t 1:00:00 # Interactive allocation"
        echo "  srun --pty bash        # Interactive shell"
        echo ""
        echo "🔍 Job Management:"
        echo "  scancel <job_id>       # Cancel job"
        echo "  scontrol show job <id> # Job details"
        echo ""
        echo "💻 Interactive Session Example:"
        echo "  srun -N 1 -t 2:00:00 --pty bash"
        echo ""
        ;;
    6)
        echo -e "\n${YELLOW}SSH Key Update Instructions:${NC}"
        echo "======================================"
        echo ""
        echo "Your current public key:"
        cat "$SSH_KEY.pub"
        echo ""
        echo "To update your SSH key with AMD:"
        echo ""
        echo "1. Copy the above public key"
        echo "2. Email AMD HPC administrators:"
        echo "   - Subject: SSH Key Update Request - pedramnj"
        echo "   - Include: Your current public key (shown above)"
        echo "   - Include: User details (pedramnj, s317086@studenti.polito.it)"
        echo ""
        echo "3. Wait for confirmation from AMD"
        echo "4. Test connection again with option 1"
        echo ""
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "For more information, see: AMD_HPC_ACCESS.md"

