import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Get a list of available voices
voices = engine.getProperty('voices')

# Print available voices
for voice in voices:
    print("Voice:")
    print(" - ID: %s" % voice.id)
    print(" - Name: %s" % voice.name)
    print(" - Languages: %s" % voice.languages)
    print(" - Gender: %s" % voice.gender)
    print(" - Age: %s" % voice.age)

# Change the voice
# For example, you can specify the voice by ID
engine.setProperty('voice', voices[1].id)  # Change to the first available voice
# Or you can specify the voice by name
# engine.setProperty('voice', 'english')  # Change to an English voice

# You can also specify the rate and volume
engine.setProperty('rate', 150)  # Change the speech rate (words per minute)
engine.setProperty('volume', 0.9)  # Change the volume (0.0 to 1.0)

# Test the voice by saying something
engine.say("Hello, what is your name.")

# Wait for the speech to finish
engine.runAndWait()
