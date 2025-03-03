# Technical Specs

## Initaization Step
When the app first initilaizes it needs to do the following:
- Fetch from the GET localhost:5000/api/groups/:id/raw, this will return a collection of words in a json structure. It will have Japaense words with the english translation. We need to store this collection of words in memory.

## Page States

Page States represents the state the single page application should behave from a users persepective. Each page state is a single page that the user can navigate to.

### Setup State
When a user first starts up the pp.
They will only see a button called "Geenrate Sentence."
when they press the button, the app will generate a sentencue using the Sentence Generator LLM, and the state will move to Practice State

### Practice State
When a user is in practice state, they will see an Engligh sentence, and also an upload field under the English sentence and will see a button called "submit for review." when they press the submit fo review button an uploaded image will be passed to the gradidng system and then will transition to the Review State.

### Review State
When a user is in Review state, the user will still see the English sentence. The upload field will be gone.The user will now ss a review of the output from the Grading System. There will be a button called "Next Question." when clicked, it will generate a new question and place the the app into Practice State.
The grading system will have the following information:
- transcription of Image
- Translation of image
- Translation of Transcription
- Grading
    -  A Letter score using the 5 Rank to score
    - A description of whether the attempt was accurate to the English sentence and suggestions.


## Senetecne Genator LLM Prompt
Generate a sentcne using the following {{word}}
The grammar should be JLPIN5
You ca use the following vocabulary to construct a simple sentecne:
- simple objects eg. book, car, rame, sushi
- simple verbs eg. to drink, toeat, to meet
- simple times eg. tomorrow, today, yesterday

## Grading System
The grading system will have the following information:
- transcription of Image It will transcribe the Image using MangoOCR
- It will use a LLM to product a little translation of the transcription
- It will use another LLM to prduce a grade
- It will tehn return this data to the frontend app