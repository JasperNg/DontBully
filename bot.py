import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import discord
import os

model = tf.keras.models.load_model(("my_model.h5"),custom_objects={'KerasLayer':hub.KerasLayer})

learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.001,
    end_learning_rate=0.0001,
    decay_steps=10000,
    power=0.5)
opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])


naughty_list =[]

client = discord.Client()


def compute(msg, message):
    output = model.predict([str(msg)])
    s = str(output)
    start = "[["
    end = "]]"
    num = float(s[s.find(start) + len(start):s.rfind(end)])
    if num > 0.5:
        return 1
    else:
        return 0

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    print('If this bullying that is not detected correctly, please @mention the moderators.')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    msg = message.content

    answer = compute(msg, message)
    if answer == 1:
        mention = message.author.mention
        au1 = message.author
        await message.channel.send(f'{mention} Bullying Detected. You will be kicked the next time it happens. If '
                                   f'this bullying that is not detected correctly, please @mention the moderators.')
        #if au1 in naughty_list:
            #await au1.kick(reason=None)
            #naughty_list.remove(au1)
            #await message.channel.send(f'{mention} was kicked')
        #else:
            #await message.channel.send(f'{mention} Bullying Detected. You will be kicked the next time it happens. If '
                                       #f'this bullying that is not detected correctly, please @mention the moderators.')
            #naughty_list.append(au1)

client.run('TOKEN')






