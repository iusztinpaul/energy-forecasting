# The Full Stack 7-Steps MLOps Framework

### 🔥 LIVE DEMO 🔥 | [WEB APP - FORECASTING](http://35.207.134.188:8501/) | [WEB APP - MONITORING](http://35.207.134.188:8502/)

--------

If you reached this far, congratulations, you are close to the full experience. This is the last step from the 7 lessons of the course.

# CI/CD

We will use GitHub Actions to create the CI/CD pipeline. GitHub Actions will let us run various commands on specific triggers, such as a new push to a branch.

## Fork the Repository

By forking the repository you will create an exact same copy of the code on your own private GitHub account. Thus, you will have full access to the settings of the repository.

[Check out this docs](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

## Set Actions Variables

Go to your forked repository. After go to `Settings` -> `Secrets and variables` (in the Security tab) -> `Actions`.

After click `Variables`. From there you can create a new variable by clicking `New repository variable`.

See the image below 👇

<p align="center">
  <img src="images/github_actions_variables_screenshot.png">
</p>

You have to create 5 variables that will be injected into the GitHub Actions scripts:
* `APP_INSTANCE_NAME` : the name of the web app VM. | In our case it is `app`. The default should be ok if you haven't changed anything.
* `GCLOUD_PROJECT` : the ID of your GCP Project | Here you have to chance it with your own ID.
* `ML_PIPELINE_INSTANCE_NAME` : the name of the ML pipeline VM. | In our case it is `ml-pipeline`. The default should be ok if you haven't changed anything.
* `USER`: the user you used to connect to the VMs while settings up the machine with SSH. | Mine was `pauliusztin`, but you have to change it with yours.
* `ZONE` : the zone where you deployed the VMs. | If you used the same as ours, you can also fill it with `europe-west3-c`

## Set Secrets

In the same sections, hit the `Secrets` tab. 

You can create a new secret by pressing the `New repository variable` button.


These are similar to the variables we just completed, but after you fill their values, you can't see them anymore. That is why these are called secrets. Here is where you add all your sensitive information. In our case the GCP credentials and private keys.

See the image below 👇

<p align="center">
  <img src="images/github_actions_secrets_screenshot.png">
</p>

The `GCP_CREDENTIALS` secret contains the content of the JSON key of your VM admin service account. By settings this up, the CI/CD pipeline will use that service account to authenticate the VMs.

Because the content of the file is in JSON format, you have to run the following command:
```shell
jq -c . /path/to/your/admin-vm.json 
```
Take the output of this command and create your `GCP_CREDENTIALS` secret with in.

The `GCP_SSH_PRIVATE_KEY` is your GCP private SSH (not your personal one - GCP creates an additional one automatically) key which was created on your local computer when you used SSH to connect to the VMs.

To copy it, run the following:
```shell
cd ~/.ssh
cat google_compute_engine
```
Copy the output from the terminal and create the `GCP_SSH_PRIVATE_KEY` variable. 


## Run the CI/CD Pipeline

Now make any change to the code, push it to git, and the GitHub Actions should trigger automatically.

To see their results check the `Actions` tab from your GitHub repository.

<p align="center">
  <img src="images/github_actions_see_cicd_screenshot.png">
</p>

Two actions will be triggered. One that will build and deploy the `ml-pipeline` modules and one that will build and deploy the `web app`. 

If you want to understand better how we wrote the GitHub Actions scripts under the `.github/workflows` directory [check out this Medium article](placeholder for Medium article) that explains everything in detail.
