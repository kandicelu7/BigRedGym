# Deploying a custom policy

## Setup

### Install required packages

To deploy a policy on the robot from your computer, you need to install the additional libraries ```cyclonedds``` and ```unitree-sdk2py```. To do this, run the following in the terminal:

```bash
uv run --frozen scripts/fetch_unitree_sdk.py
```

Then run this:

```bash
mkdir -p "$HOME/.local/src"
git clone --depth 1 --branch 0.10.2 \
    https://github.com/eclipse-cyclonedds/cyclonedds.git \
    "$HOME/.local/src/cyclonedds-0.10.2"
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
cmake -S "$HOME/.local/src/cyclonedds-0.10.2" \
    -B "$HOME/.local/src/cyclonedds-0.10.2/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build "$HOME/.local/src/cyclonedds-0.10.2/build" \
    --config RelWithDebInfo --target install --parallel 4
```

Then run this:

```bash
uv sync --frozen --extra unitree_sdk
```

The output should be something along the lines of:

```
Built unitree-sdk2py @ <location>
Built cyclonedds==0.10.2
...
```

### Set up ethernet connection

Instructions are for Linux (Ubuntu)
- Connect an ethernet cable to the robot and to your device.
- Go to Settings -> Network -> click the settings icon under Wired -> IPv4
- Switch IPv4 Method to Manual
- Under Addresses, in the first row, type `192.168.123.99` under Addresses, and `255.255.255.0` under Netmask
- Click Apply
- Turn on the connection under Wired

## Configure deployment

Settings can be configured in `go2_deploy/deploy_config.py`. Options include:
- Control frequency
- kp (Stiffness constant)
- kd (Damping constant)
- Observation vector: a list of available observations can be found in `obs_sizes`
- Command limits (ex. pushing the joystick all the way to the right commands x m/s)
- Gait frequency (`phase_frequency`, in Hz): controls the Go2Trot phase clock and its frequency observation

For current Go2Trot policies, joint-position observations are relative to the
static nominal posture from `Go2TrotCfg` (hip 0, thigh 0.96, calf -1.36 radians).
Scaled policy actions are offsets from that posture. The gait reference is a
training reward only; deployment retains the sine/cosine phase observations and
observes the controller's active gait frequency, divided by its observation scale.
The previous-action observation contains the applied offset after optional EMA
smoothing. Neither that offset nor the resulting position target is clipped.
Deployment sends position targets and PD gains to Unitree; this path does not
compute or clamp torques locally.

Offline utility tests cover these mappings using synthetic messages. They do
not establish hardware behavior or compatibility with an older checkpoint's
controller semantics.

## Deployment

If anything goes wrong, the emergency stop on the wireless controller is X or the enter key in the terminal.

- Make sure your trained policy is in `logs/`, and that the observation vector, scaling, and other settings in `deploy_config.py` are all correct.
- Turn on the Go2 and wait until it is in the standing position.
- In the terminal, run `uv run --extra unitree_sdk python -m go2_deploy.deploy eth0`, but replace `eth0` with the actual network configuration. Keep `CYCLONEDDS_HOME` exported. If this is successful, there should be a terminal output every few seconds telling you about the keyboard controls, observation frequency, etc.
- There are 4 states the robot can be in:
    - `RECOVERY`: Initially when the script starts running, the robot is in this mode. You must enter this state from `EMERGENCY_STOP` otherwise. The default Unitree controller can be used to make the robot walk, do tricks, etc. The `F1` key on the remote control can be used to disable/enable the A, B, X, and Y buttons, so that the default Unitree button combinations can be used without accidentally turning on a different state.
    - `EMERGENCY_STOP`: The robot will immediately start damping, wait around 15 seconds, and then stand up again. The robot will automatically enter the `RECOVERY` state.
    - `INTERMEDIATE`: The robot will hold its current position. This is for testing to make sure the kp and kd values are good, etc.
    - `CUSTOM_CTRL`: The robot will use the policy you trained to move.
- To stop running the script, return to `RECOVERY`, lie the robot on the ground, and press `Ctrl+C` in the terminal.
- Logs of the robot data can be found in `logs/deploy/`.

## Remote controls

- Switch modes
    - Enter `EMERGENCY_STOP`: `X`
    - Enter `INTERMEDIATE`: `B`
    - Enter `CUSTOM_CTRL`: `A`
- Disable/enable ABXY buttons: `F1`
- Increase kp by 10%: `Up` (below the left joystick)
- Decrease kp by 10%: `Down`
- Increase kd by 10%: `Right`
- Decrease kd by 10%: `Left`

## Keyboard controls
Type the following letters in the terminal:

- Switch modes
    - Enter `EMERGENCY_STOP`: `<enter key>`
    - Enter `INTERMEDIATE`: `q`
    - Enter `CUSTOM_CTRL`: `c`
- Increase kp by 10%: `i`
- Decrease kp by 10%: `k`
- Increase kd by 10%: `l`
- Decrease kd by 10%: `j`

## Something went wrong

- `deploy.py` is printing errors in the terminal
    - Make sure the robot is either lying down, or standing up while in sport mode, and exit the script in the terminal.
- The robot is shaking violently when running a custom policy
    - In `deploy_config.py`, make sure that your observation vector is correct and the values in the `DeployScaling` class match what you used during training. Also make sure you are running the correct policy from `logs/`.
