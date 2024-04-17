import numpy as np
import cv2
import torch
import improc
import time
import pyduino
import win32api
import csv
from csv import reader
import pandas as pd
import os
import winsound
from threading import Thread


def setpriority(pid=None,priority=1):
    """ Set The Priority of a Windows Process.  Priority is a value between 0-5 where
        2 is normal priority.  Default sets the priority of the current
        python process but can take any valid process ID. """

    import win32api,win32process,win32con

    priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                       win32process.BELOW_NORMAL_PRIORITY_CLASS,
                       win32process.NORMAL_PRIORITY_CLASS,
                       win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                       win32process.HIGH_PRIORITY_CLASS,
                       win32process.REALTIME_PRIORITY_CLASS]
    if pid == None:
        pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, priorityclasses[priority])

class ArduinoHandler(object):
    """ for handling the communication with the arduino (for synchronization of mvn xsens recording) """
    def __init__(self,serial_port='COM11',LED = 13, SYNC = 12):
        self.connected = False
        self.arduino=None
        self.serial_port = serial_port
        self.LED = LED
        self.SYNC = SYNC
        self.state = 0
        while not self.connected:
            try:
                self.arduino = pyduino.Arduino(serial_port)
                time.sleep(2)
                self.arduino.set_pin_mode(LED,'O')
                time.sleep(1)
                print("succesfully connected with arduino")
                for i in range(10):
                    if i%2 == 0:
                        self.arduino.digital_write(LED,1) # turn LED on
                    else:
                        self.arduino.digital_write(LED,0) # turn LED off
                time.sleep(0.1)
                self.connected = True
                break
            except:
                print("Could not find arduino on {!r}".format(serial_port))
                serial_port=input("Try again on serial port: ")
                if not serial_port:
                    print("WARNING XSens MVN will not be able to start synchronized recording!")
                    break
    def send_pulse(self):
        if self.arduino is not None:
            self.state = 1-self.state
            self.arduino.digital_write(self.SYNC,self.state) # turn SYNC on
            self.arduino.digital_write(self.LED,self.state) # turn LED on
            # time.sleep(.1)
            # self.arduino.digital_write(self.SYNC,0) # turn SYNC off
            # self.arduino.digital_write(self.LED,0) # turn LED on
        else:
            print("WARNING Arduino not connected! No XSens MVN recording is started/ended!")
        return

class Subject(object):
    """Class for loading the subject-specific randomization protocol, and loading/saving
     the recorded timestamps """
    def __init__(self,ID, randomizationKey='./randomization/randomization_key_191120.csv'):
        # Load previously randomized trial protocol for current subject
        self._ID                = ID
        self._randomizationKey  = pd.read_csv(randomizationKey)
        self._protocol          = self._randomizationKey[self._randomizationKey.participant == ID].set_index('trial')
        print("Found {} trials for participant {}".format(len(self._protocol),ID))
        assert len(self._protocol)>0

        # For saving the timestamps
        self._rootdir           = '../Data/{}/'.format(ID)
        self._eventsdir         = os.path.join(self._rootdir,'Events')
        self._videodir          = os.path.join(self._rootdir,'Videos')
        self._filename          = os.path.join(self._eventsdir,'{}_timestamps.csv'.format(ID))
        self.results            = None

        # Scan directory and look for previously saved timestamps
        if not os.path.isdir(self._eventsdir):
            print('creating directory {}'.format(self._eventsdir))
            os.makedirs(self._eventsdir)
        if not os.path.isdir(self._videodir):
            print('creating directory {}'.format(self._videodir))
            os.makedirs(self._videodir)

        # Load if previously recoded timestamps exists
        if os.path.exists(self._filename):
            self.loadResults()
        else:
            self.makeResults()

    def makeResults(self):
        self.results                    = self._protocol.copy()
        self.results['events']          = ''
        self.results['trial_duration']  = ''
        self.results['framerate']       = ''
        self.results['rating']          = ''
        self.results['comments']        = ''
        self.saveResults()

    def loadResults(self):
        print('ATTENTION filename already exists! Loading previously recorded timestamps')
        self.results = pd.read_csv(self._filename).set_index('trial').fillna('')
        assert len(self.results) == len(self._protocol)
        print(self.results)

    def saveResults(self):
        print('Saving results to {}'.format(self._filename))
        self.results.to_csv(self._filename)

class Trial(object):
    """ Class for handling (e.g. starting/ending) individual trials.
    note: the class uses global variables 'results', 'filename' and 'arduino'"""
    def __init__(self,subject,idx):

        self.active     = False
        self.nFrames    = 0

        # Get the subject-specific trial protocol for corresponding trial
        self.subject    = subject
        self.idx        = idx
        self.specs      = subject.results.loc[idx].copy()

        # Phosphene simulation
        self.simulation = self.specs.condition_1
        self.mode       = {'CA':0, 'CE':1, 'SN':2}[self.simulation[:2]]
        self.resolution = {'1':(10,10), '2':(18,18), '3':(26,26),
                           '4':(34,34), '5':(42,42), '6':(50,50),
                           'M':None}[self.simulation[-1]]
        # Results
        self.events     = []
        self.duration   = ''
        self.frames     = []

        print('\nPress ENTER to start trial {} (session {session}) on {condition_3}'.format(idx, **self.specs))
    def start(self):
        self.active     = True
        self._started   = time.time()
        winsound.Beep(frequency=1000,duration=200)
        return self
    def event(self):
        t = time.time()-self._started
        print('event at {:.2f}s'.format(t))
        self.events.append(t)
        return t
    def end(self):
        self.duration = time.time()-self._started
        self.framerate = min([30,self.nFrames/self.duration])
        winsound.Beep(frequency=1000,duration=500)

        arduino.send_pulse()

        # Save trial specs
        self.rating = rate_trial()
        self.specs.update({'trial_duration':  '{:.2f}'.format(self.duration),
                           'events': ' '.join(['{:.2f}'.format(t) for t in self.events]),
                           'framerate': '{:.2f}'.format(self.framerate),
                           'rating' : '{}'.format(self.rating)})

        self.subject.results.loc[self.idx] = self.specs
        print('Trial duration: {trial_duration} | Framerate: {framerate} | Events: {events} | Rating: {rating}'.format(**self.specs))
        self.subject.saveResults()

        # Save video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = os.path.join(subject._videodir, '{}-{:02d}.avi'.format(subject._ID,self.idx))  #{'subject':subject._ID, 'trial':self.idx}))
        out = cv2.VideoWriter(filename, fourcc, 10, (960,480))
        print('Saving video to {}'.format(filename))
        for i,f in enumerate(self.frames):
            out.write(f)
        out.release



        self.active = False
        return self.specs

    def next(self,idx=None):
        idx = self.idx + 1 if idx is None else idx
        if idx <= len(self.subject.results):
            self.__init__(subject=self.subject, idx=idx)
        else:
            print('Finished all trials!! :)')
            idx = input('Re-do specific trial? Please enter trialindex below. (Otherwise leave empty)\nidx:')
            if idx:
                self.__init__(subject=self.subject, idx=idx)
            else:
                exitSoftware()
        pass

def rate_trial():
    print("\nplease indicate subjective trial rating on scale 1 to 10 (press 0 for 10)\n")
    key = None
    while key is None or key < 48 or key > 57:
        frame = camera.frame
        display.frame = improc.crop_resize(frame)
        key = display.getKey()
    rating = key-48
    return rating





def exitSoftware():
    print("Quitting...")
    display.stop()
    camera.stop()
    cv2.destroyAllWindows()
    quit()


# initialize subject (load the randomized trial protocol and previously saved timestamps)
ID=input("Please enter participant ID (e.g. HC01): ")
subject = Subject(ID=ID, randomizationKey='./randomization/randomization_key_040221.csv')#randomization_key_251120.csv')

# arduino (for synchronized mvn xsens recording)
arduino = ArduinoHandler(serial_port='COM11',LED = 13, SYNC = 12)

# preprocessing models
sn_filter = improc.SharpNetFilter(checkpoint_path='./model/weights/final_checkpoint_NYU.pth',
                                device='cuda:0',smooth_output=1.5,threshold_b=94, threshold_c=70)
ced_filter = improc.CannyFilter(sigma=3,low=25,high=50)
simulator = improc.PhospheneSimulator(intensity=3)


# Webcam
cameradevice = 1 #VR webcam
camera =  improc.VideoGet(cameradevice)
if camera.frame is None:
    print("WARNING could not detect frontcam! Is the headset properly connected?")
    camera.stop()
    camera.__init__(cameradevice=0)
camera.start()



# VR Presentation (runs in another seperate thread)
display = improc.ShowVR(windowname='VRSimulation',resolution=(1600,2880), imgsize=(480,480), ipd=1240)
display.start()

# Interaction with windows system
setpriority(priority=4) #set high priority for python process


mode = 0
trial = Trial(subject=subject, idx=1)


t1 = int((10*time.time())%10)

# Video processing
while not camera.stopped or display.stopped:

    # Grab frame
    trial.nFrames = trial.nFrames+1
    frame = camera.frame


    # Do pre-processing and phosphene simulation
    if mode == 0: # Just camera
        frame = improc.center_crop(frame, resize=(480,480), zoom=1.9)
    elif mode == 1:
        frame = improc.center_crop(frame, resize=(480,480), zoom=1.9)
        frame = simulator(ced_filter(frame))
    elif mode == 2:
        frame = improc.center_crop(frame, zoom=1.9)
        frame = simulator(sn_filter(frame, resize=(480,480)))

    # Display on VR display
    display.frame = frame

    # Every 10s append frame
    t2 = int((10*time.time())%10)
    if trial.active and t2!=t1:
        t1 = t2
        trial.frames.append(np.concatenate([improc.crop_resize(camera.frame), frame], axis=1))


    # Check for keypress
    key = display.getKey()


    if key is not None:
    #Key pressed

        if key == 13: ## ENTER -> start/end trial
            if trial.active:
                specs = trial.end()
                trial.next()
                mode = 0
            else:
                n_frames = 0
                print('______________________________________________________________________________')
                print('Trial {}\n'.format(trial.idx))
                if trial.mode != 0:
                    simulator = improc.PhospheneSimulator(phosphene_resolution=trial.resolution, intensity=3)
                mode = trial.mode
                trial.nFrames=0
                if mode ==2: # SharpNet mode
                    _ = sn_filter(np.zeros((480,480,3))) # Parse dummy frame (for speeding up)
                arduino.send_pulse()
                trial.start()


        elif key == 32 and trial.active: #SPACEBAR
            trial.event()

        elif key == 45 and not trial.active:
            print('Going back')
            idx = max([1,trial.idx - 1])
            trial.next(idx=idx)

        elif key == 61 and not trial.active:
            print('Going forward')
            idx = min([len(subject.results), trial.idx + 1])
            trial.next(idx=idx)

        elif key == ord('p'):
            print("sending pulse..")
            t2 = Thread(target=arduino.send_pulse())
            t2.start()

        elif key == ord('q'):
            exitSoftware()



exitSoftware()
