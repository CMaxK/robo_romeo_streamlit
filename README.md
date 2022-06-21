# Data analysis
- Document here the project: robo_romeo_streamlit
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for robo_romeo_streamlit in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/robo_romeo_streamlit`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "robo_romeo_streamlit"
git remote add origin git@github.com:{group}/robo_romeo_streamlit.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
robo_romeo_streamlit-run
```

# Install

Go to `https://github.com/{group}/robo_romeo_streamlit` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/robo_romeo_streamlit.git
cd robo_romeo_streamlit
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
robo_romeo_streamlit-run
```
# Website

Link : [https://awesome-github-readme-profile.netlify.app](https://share.streamlit.io/cmaxk/robo_romeo_streamlit/app.py)
