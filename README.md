# Sup? 🧠 📸🖼️

Everyone likes AI/ML/GLSL so let's have an AI/ML/GLSL party up in here! 🎉🥳🎊🎁

I'm using 3.10.10 with pyenv. You should find all goodies in requirements.txt

See it in action on [shadertoy#WXsGWN](https://www.shadertoy.com/view/WXsGWN), the loss there was under .004.
And took about an hour to train.

## (Quick) Training Example 🏋️‍♂️

This is a quick run to show for now: ![training output image](run/training-2025-03-26_19-59-54/images/output.png) which only ran a few minutes with a loss of 0.006454

![training](run/training-2025-03-26_19-59-54/training.gif)

I'm running a longer training now (about 2 hours) to see if more time will make much difference.

After that I will try more experiments with the training code itself probaby following a few ideas:
1. better image pre-processing
2. varying the test inputs
3. changing around the loss function
4. tweaks to the training method 
5. try it on different images

# Train the model 🏋🏽🔥💪🏼🎧

All you need is the input image. Here is an example specifying some output files as well

```
time ./futuristica.py --weights lenna.npz --generated output.png --training 30 --image lenna.png 
```

On my rtx 3060, training for 30 (x 5000) epochs takes about 30 minutes.

I strongly recommend using `run-training.sh` to keep thing organized, see below.

## Stuck Training 😩😮‍💨

If you start to see a line like this: `Epoch 265161: PUNT: 5001 is too long!`, then training is probably stuck. 
You'll either need to tweak the model or the code or just reroll and hope the RNG loves you better.

During training it will probably drop png's and weights.npz files so you should be ok to kill it but find the
weights file first!


# Generate GLSL for ShaderToy 🌗🧸🧩🚂

This will dump a big bunch spew: `./translate.py lenna.npz`

# Tweak the model 🛠️🧠💡🤓🤔💪🏻

Edit futuristica.py and change the definition of layers in the MLP class at the top.

# Helper Scripts 🤝🤗🛟🚢🆘🛟🚨📢

The script `run-training.sh` helps save output from training runs into
a directory for each run.  It's there to help keep things organized.

Each training will have it's output in a timestamped directory under
./run, just trying running it... Some info bits might not work until
you deal with the dependencies, but training should still work.

Note: it will also create a symbolic link from the main directory to
the latest training run for convenience / laziness.

The script `historic-output.sh` will 
- look at the output png's from training
- display the latest one
- create an mp4 of all of them
- display the mp4 as a time lapse
I wrote it cuz it's nifty.

## Script Dependencies 🔗⚓

To run the support scripts you'll need some tools installed or to edit them to use your preffered

I use an ancient, crusty program called "xv" which you can replace with any thing else that can display a png.

The `historic-output.sh` script has a bit more odds and ends in addition to "xv":
- ffmpeg, to create an mp4 from the training output videos
- mplayer to display the mp4
