## This repo is for the project of our last year of engineering in AI.

Our project subject is the analysis of french political debate using different AI technologies.

We did STT (Speach To Text) to transcript all the text using a trained model.
We trained a model to recognize who talks.

### The feature we did are:
  - STT
  - Sentiment analysis
  - Voice recognition
  - Temporal parity
  
## File architecture

You will find different folder:
  - tools -- py or ipynb to either convert file or record voice/label
  - data -- empty but with a link to the data we used if you want to retrain your models
  - models -- different models in py or ipynb
  - GUI_voice_recording_text_label -- little GUI to record and generate a text/wav with the same name

## Credits

We used AudioMNIST, ATT-HACK repo and research paper to do our projects.
Links are below
1. ATT-HACK
   - REPO: https://gitlab.com/nicolasobin/att-hack/-/blob/master/README.md
   - PAPER: https://arxiv.org/pdf/2004.04410.pdf
   - DATA: https://openslr.org/88/
2. AudioMNIST
   - REPO: https://github.com/soerenab/AudioMNIST
   - PAPER: https://arxiv.org/pdf/1807.03418.pdf
   - DATA: Inside the repo
