# #!/usr/bin/env bash
# set -ex

# # This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
# echo "Sorry, we couldn't automatically determine the right command for this file type. Please replace this line with a command to run 'run'."

# The previous version of this file was commented-out and follows below:
#
#!/usr/bin/env bash
set -ex


echo "Starting the reproducible run script..."


echo "Upgrading metagpt..."
pip install --upgrade metagpt

echo "Setting up SSH port forwarding..."

# If you need to enter the password automatically, install sshpass
if ! command -v sshpass &> /dev/null; then
    echo "Installing sshpass..."
    apt-get update && apt-get install -y sshpass
fi


REMOTE_HOST="connect.cqa1.seetacloud.com"
REMOTE_PORT="19139"
REMOTE_USER="root"
LOCAL_PORT="6006"
REMOTE_PORT_FORWARD="127.0.0.1:6006"


PASSWORD="7pG33QeFOYAc"

if ! command -v sshpass &> /dev/null; then
    echo "Installing sshpass..."
    sudo apt-get update && sudo apt-get install -y sshpass
fi

# SSH
sshpass -p "$PASSWORD" ssh -CNg -L $LOCAL_PORT:$REMOTE_PORT_FORWARD $REMOTE_USER@$REMOTE_HOST -p $REMOTE_PORT &


sleep 5

echo "SSH port forwarding set up successfully."

# This toy script is used to present the conversation of clinician agents.
echo "Running MAgent.py..."
python /code/examples/clinic/MAgent.py



# 
echo "All tasks completed successfully."
