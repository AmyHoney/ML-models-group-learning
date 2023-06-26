## Speech2Text Wav2Vec2 example:
In this example we will use a pretrained Wav2Vec2 model for Speech2Text using the `transformers` library: https://huggingface.co/docs/transformers/model_doc/wav2vec2 and serve it using torchserve.

**If you want to know more details, please see speech2text_wav2vec2.ipynb**

### Create and run a Notebook server

The dependencies have integrated into the below docker image, you can directly use an image published on VMware harbor repo to run speech2text model:

```bash
projects.registry.vmware.com/models/notebook/hf-inference-deploy:v1
```

**No need GPU for speech2Text Wav2Vec2 model deployment** 

### Prepare model and run server
Next, we need to download our wav2vec2 model and archive it for use by torchserve:
```bash
./download_wav2vec2.py # Downloads model and creates folder `./model` with all necessary files
./archive_model.sh # Creates .mar archive using torch-model-archiver and moves it to folder `./model_store`
```

Now let's start the server and try it out with our example file!
```bash
torchserve --start --model-store model_store --models Wav2Vec2=Wav2Vec2.mar --ncs
# Once the server is running, let's try it with:
curl -X POST http://127.0.0.1:8080/predictions/Wav2Vec2 --data-binary '@./sample.wav' -H "Content-Type: audio/basic"
# Which will happily return:
I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT%
```
