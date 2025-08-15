### Project Roadmap
1. [x] Environment: Define Dockerfile, docker-compose.yml, Makefile.
2. [ ] Predictive Engine (train_model.py):
  . [ ] Create synthetic data generator.
  . [ ] Integrate PyCaret to train and save the model.
3. [ ] Prescriptive Engine (genai_prescriptions.py):
  . [ ] Define a robust prompt for JSON output.
  . [ ] Implement API calls for GenAI services.
4. [ ] UI & Orchestration (app.py):
  . [ ] Build input sidebar.
  . [ ] Orchestrate the prediction/prescription workflow.
  . [ ] Design output tabs for results.
