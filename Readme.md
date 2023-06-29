# Twibot

Here is my Twibot project.

The aim of this project is to make a bot able to play a game named Twinit using AI and some image processing techniques.

## What is Twinit?

Twinit is a game where the goal is to be the faster among your opponents to point out two identical cards. 
The cards are arranged on the table, visible to all. 

[Here](https://www.youtube.com/watch?v=-f1PxF9CDfU) is a detailled explanation of the game (in french).

## Steps of the project

I have broken down the project in two steps.

- Detect the cards
- Match identical cards

### Detect the cards

For this step there are no challenges that are insurmountable. 
The longest part was to build a dataset and annotate a sufficient number of cards to fine tune a YOLO.
I have limited the task to detect cards on a black background to make things easier.
Because the task is quite easy, I fine-tuned the smallest model of YOLOv7.