# Configuration 

> Instructions for setting your system at SACLA servers 

-----------------------------
### Setting up your VPN 

Follow intructions here: 
http://xfel.riken.jp/users/pdf/hpc_startup_manual_ver1.8.pdf 

Once you have intalled the Cisco Secure Client connect to: 
```bash
hpc.spring8.or.jp
```

-----------------------------

### Connecting and transfering 

Here replace "fperakis" with your user name.
```bash
ssh -X fperakis@xhpcfep.hpc.spring8.or.jp
```

```bash
scp local_file fperakis@xhpcfep.hpc.spring8.or.jp:/home/fperakis
```

-----------------------------
### Setting up procedure 

activate SACLA tools
```bash
ml python/SACLA_python-3.7/offline
which python
```

create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate
which python
```

add to your ~/.bashrc the following 
```bash
source venv/bin/activate
export PIP_PROXY=http://proxy.hpc.spring8.or.jp:3128
export HTTPS_PROXY=http://proxy.hpc.spring8.or.jp:3128
```

-----------------------------
### Running Jupiter lab

run on terminal on remote computer (here port number 8887 is chosen randonmly)
```bash
jupyter lab --no-browser --port 8887 &
```

run on new terminal on local computer 
```bash
ssh -N -L localhost:8887:localhost:8887 fperakis@xhpcfep.hpc.spring8.or.jp
```

open in browser
http://localhost:8887

copy paste “token” and you are good to go.

-----------------------------

### setting up GitHub repo
#### configure 
```bash
git config --global user.name fperakis
git config --global color.ui auto
git config --global user.email f.perakis@fysik.su.se
git config --global http.proxy http://proxy.hpc.spring8.or.jp:3128
```

#### standard git commands
```bash
git clone https://github.com/fperakis/SACLA_2025A8039.git
git status
git add [file]
git commit -m “[message]”
git push
git pull
```
note that to be able to push you will need to create a PAT (instead of password), I used the classic token
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token 
 

#### working with branches. 
Here we use for example the branch includeTagNumbers

list existing branches
```bash
git branch
```
Create a new branch
```bash
git checkout -b origin/includeTagNumbers
```
Move to a branch
```bash
git checkout origin/includeTagNumbers
```

Pull from a branch
```bash
git pull origin includeTagNumbers
```

Push to a branch
```bash
git push origin head/orgin/includeTagNumbers
```
 
