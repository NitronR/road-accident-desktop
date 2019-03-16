import kivy
kivy.require('1.10.1')

# kivy imports for UI
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.videoplayer import Video
from kivy.uix.label import Label

# fastai imports
from fastai import *
from fastai.vision import *

# other imports
import threading
import requests

# # Setup model
classes = ['accident', 'not_accident']

# data attributes
size = 299

data = ImageDataBunch.single_from_classes("", classes, ds_tfms=get_transforms(), size=size).normalize(imagenet_stats)

root = Path('D:\Programming\Machine_Learning\Projects\Road Accident Detection')

# model attributes
model_type = models.resnet18
model_dir = root
model_file = 'ra-18-1-prot'

learn = create_cnn(data, model_type, path=model_dir)
learn.load(model_file)

# app settings
video_file = 'clip1.mp4'
sample_interval = 2


class AlertThread(threading.Thread):

    def __init__(self, ac_fl_name):
        self.ac_fl_name = ac_fl_name
        threading.Thread.__init__(self)

    def run(self):
        # TODO cam_num
        r = requests.post("http://127.0.0.1:8000/submit_ra", data={'camera_num': '12524'}, files={'image': open(f'tmp/{self.ac_fl_name}', 'rb')})
        print(r.status_code, r.reason)


# Prediction thread
class PredictThread(threading.Thread):

    def __init__(self, label):
        self.label = label
        threading.Thread.__init__(self)

    def run(self):
        # load frame and predict
        frame = open_image('tmp/frame.jpg')
        frame.resize(size)

        pred = learn.predict(frame)

        # self.label.text = str(pred[0])
        print(pred)

        if(str(pred[0]) == 'accident'):
            frame.save('tmp/accident.jpg', flipped=False)
            AlertThread('accident.jpg').start()


# Setup app
class RADApp(App):

    def build(self):
        root = FloatLayout()

        # setup video player
        self.player = Video(source=f'videos/{video_file}', size_hint=(0.6, 0.6),
        pos_hint={'x':0.2, 'y':0.2}, state='play')

        self.player.bind(position=self.on_pos_change())
        root.add_widget(self.player)

        # setup label
        self.label = Label(pos_hint={'x':0, 'y':-0.4})
        root.add_widget(self.label)

        # time until next sample
        self.next = 0
        self.f_num = 1

        return root

    def on_pos_change(self):

        def handler(instance, value):
            if(value > self.next and self.player.texture is not None):
                print(value)
                self.next = value + sample_interval

                # save video frame
                self.player.texture.save('tmp/frame.jpg', flipped=False)

                # predict
                # PredictThread(self.label).start()
                
                frame = open_image('tmp/frame.jpg')
                frame.resize(size)

                pred = learn.predict(frame)

                # self.label.text = str(pred[0])
                print(pred)

                if(str(pred[0]) == 'accident'):
                    self.player.texture.save('tmp/accident.jpg', flipped=False)
                    AlertThread('accident.jpg').start()

                print()

        return handler


if __name__ == '__main__':
    app = RADApp()
    app.run()