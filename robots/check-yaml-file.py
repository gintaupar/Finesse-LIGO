# %%
# The idea of this script is to take any changes in the YAML files in the
# commissioning repo and keep finesse-ligo upto date with them by submitting a
# merge request.
# 
# This bot will use the branch `bot-update-yamls` on finesse-ligo to commit to
# and request merges from into main.
import gitlab
import os
import git
import shutil
import time
from git import Repo
import os.path
import sys

DEBUG = False
# %%
print("Checking if current yamls are up to date with finesse-ligo...")

CURRENT_REPO = repo = Repo('.')

# if not DEBUG:
#     print("DEBUG set to true and this won't run on branches that aren't main")
#     sys.exit(1)

BRANCH = 'bot-update-yamls'
REPO_PATH = '.finesse-ligo-tmp'
COPY_FILES = [
    ('./LHO/yaml/lho_O4.yaml', REPO_PATH+'/src/finesse_ligo/parameter_files/lho_O4.yaml'),
    ('./LLO/yaml/llo_O4.yaml', REPO_PATH+'/src/finesse_ligo/parameter_files/llo_O4.yaml'),
]
if not os.path.exists(REPO_PATH):
    repo = Repo.clone_from(f'https://oauth2:{os.environ['GIT_PASSWORD']}@git.ligo.org/finesse/finesse-ligo.git', REPO_PATH)
else:
    repo = Repo(REPO_PATH)

origin = repo.remote(name='origin')

try:
    branch = repo.git.checkout(BRANCH)
    repo.git.pull()
    print(BRANCH, "exists on remote, updating")
except git.GitCommandError:
    branch = repo.git.checkout('-b', BRANCH)
    print(BRANCH, "does not exist on remote")

for src, dst in COPY_FILES:
    shutil.copy(src, dst)

changedFiles = [item.a_path for item in repo.index.diff(None)]

for item in changedFiles:
    print("Changed files", item)
    
if len(changedFiles) == 0:
    print("No YAMLs were changed relative to finesse-ligo")
else:
    print("Pushing changes to", BRANCH)
    repo.index.add(changedFiles)
    repo.index.commit(f"{repo.head.commit.author.name} <{repo.head.commit.author.email}> {CURRENT_REPO.head.object.hexsha} at {time.asctime(time.gmtime(repo.head.commit.committed_date))}\n\n{CURRENT_REPO.head.object.message}")
    repo.git.push("--set-upstream", "origin", BRANCH)
    
    # a commissioning-robot token has been made for this purpose, the CI should
    # have a hidden variable it uses
    gl = gitlab.Gitlab('https://git.ligo.org', private_token=os.environ['commissioning_robot'])
    
    gl.auth()
    project = gl.projects.get('finesse/finesse-ligo')

    mrs = project.mergerequests.list(state='opened', source_branch=BRANCH)
    print("Open merge requests", mrs)
    
    # If nothing is open we make a new one
    if len(mrs) == 0:
        DESC = """This is an automated merge request to update the site YAML files stored in
        finesse-ligo with those on the https://git.ligo.org/IFOsim/ligo-commissioning-modeling
        repository.
        """
        
        if DEBUG:
            DESC = "DEBUGGING don't merge!\n\n" + DESC
        
        mr = project.mergerequests.create({
            'source_branch': BRANCH,
            'target_branch': 'main',
            'title': 'Auto-bot update site YAMLs from commissioning repository',
            'description': DESC
        })
        
        print("Created merge request: ",  mr.web_url)