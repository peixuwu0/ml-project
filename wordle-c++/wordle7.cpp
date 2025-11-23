#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <stdexcept>
//Trisha Wu
using namespace std;

struct Guess {
    string word;
    string feedback;
};

class WordleGame {
private:
    vector<string> wordList;
    string secretWord;
    int attempts;

public:
    WordleGame();

    static int generateRandomIndex(int maxIndex);

    static bool isValidInput(const string& input);

    static void provideFeedback(const string& secretWord, const string& guess, Guess& result);

    void playGame();
};

// Input:None Pw
//Process:Iinitialize variables, create a list of words, select a random word as secret word.PW
//Output:None PW
WordleGame::WordleGame() : attempts(0) {
    srand(static_cast<unsigned int>(time(0)));
    wordList = {
 "about", "alert", "argue", "beach",
 "above", "alike", "arise", "began",
 "abuse", "alive", "array", "begin",
 "actor", "allow", "aside", "begun",
"acute", "alone", "asset", "being",
"admit", "along", "audio", "below",
"adopt", "alter", "audit", "bench",
"adult", "among", "avoid", "billy",
"after", "anger", "award", "birth",
"again", "angle", "aware", "black",
"agent", "angry", "badly", "blame",
"agree", "apart", "baker", "blind",
"ahead", "apple", "bases", "block",
"alarm", "apply", "basic", "blood",
"album", "arena", "basis", "board",
"boost", "buyer", "china", "cover",
"booth", "cable", "chose", "craft",
"bound", "calif", "civil", "crash",
"brain", "carry", "claim", "cream",
"brand", "catch", "class", "crime",
"bread", "cause", "clean", "cross",
"break", "chain", "clear", "crowd",
"breed", "chair", "click", "crown",
"brief", "chart", "clock", "curve",
"bring", "chase", "close", "cycle",
"broad", "cheap", "coach", "daily",
"broke", "check", "coast", "dance",
"brown", "chest", "could", "dated",
"build", "chief", "count", "dealt",
"built", "child", "court", "death",
"debut", "entry", "forth", "group",
"delay", "equal", "forty", "grown",
"depth", "error", "forum", "guard",
"doing", "event", "found", "guess",
"doubt", "every", "frame", "guest",
"dozen", "exact", "frank", "guide",
"draft", "exist", "fraud", "happy",
"drama", "extra", "fresh", "harry",
"drawn", "faith", "front", "heart",
"dream", "false", "fruit", "heavy",
"dress", "fault", "fully", "hence",
"drill", "fibre", "funny", "night",
"drink", "field", "giant", "horse",
"drive", "fifth", "given", "hotel",
"drove", "fifty", "glass", "house",
"dying", "fight", "globe", "human",
"eager", "final", "going", "ideal",
"early", "first", "grace", "image",
"earth", "fixed", "grade", "index",
"eight", "flash", "grand", "inner",
"elite", "fleet", "grant", "input",
"empty", "floor", "grass", "issue",
"enemy", "fluid", "great", "irony",
"enjoy", "focus", "green", "juice",
"enter", "force", "gross", "joint",
"judge", "metal", "media", "newly",
"known", "local", "might", "noise",
"label", "logic", "minor", "north",
"large", "loose", "minus", "noted",
"laser", "lower", "mixed", "novel",
"later", "lucky", "model", "nurse",
"laugh", "lunch", "money", "occur",
"layer", "lying", "month", "ocean",
"learn", "magic", "moral", "offer",
"lease", "major", "motor", "often",
"least", "maker", "mount", "order",
"leave", "march", "mouse", "other",
"legal", "music", "mouth", "ought",
"level", "meant", "never", "party",
"peace", "power", "radio", "round",
"panel", "press", "raise", "route",
"phase", "price", "range", "royal",
"phone", "pride", "rapid", "rural",
"photo", "prime", "ratio", "scale",
"piece", "print", "reach", "scene",
"pilot", "prior", "ready", "scope",
"pitch", "prize", "refer", "score",
"place", "proof", "right", "sense",
"plain", "proud", "rival", "serve",
"plane", "prove", "river", "seven",
"plant", "queen", "quick", "shall",
"plate", "sixth", "stand", "shape",
"point", "quiet", "roman", "share",
"pound", "quite", "rough", "sharp",
"sheet", "spare", "style", "times",
"shelf", "speak", "sugar", "tired",
"shell", "speed", "suite", "title",
"shift", "spend", "super", "today",
"shirt", "spent", "sweet", "topic",
"shock", "split", "table", "total",
"shoot", "spoke", "taken","touch", 
"short", "sport", "taste", "tough",
"shown", "staff", "taxes", "tower",
"sight", "stage", "teach", "track",
"since", "stake", "teeth", "trade",
"sixty", "start", "texas", "treat",
"sized", "state", "thank", "trend",
"skill", "steam", "theft", "trial",
"sleep", "steel", "their", "tried",
"slide", "stick", "theme", "tries",
"small", "still", "there", "truck",
"smart", "stock", "these", "truly",
"smile", "stone", "thick", "trust",
"smith", "stood", "thing", "truth",
"smoke", "store", "think", "twice",
"solid", "storm", "third", "under",
"solve", "story", "those", "undue",
"sorry", "strip", "three", "union",
"sound", "stuck", "threw", "unity",
"south", "study", "throw", "until",
"space", "stuff", "tight", "upper",
"upset", "whole", "waste", "wound",
"urban", "whose", "watch", "write",
"usage", "woman", "water", "wrong",
"usual", "train", "wheel", "wrote",
"valid", "world", "where", "yield",
"value", "worry", "which", "young",
"video", "worse", "while", "youth",
"virus", "worst", "white", "worth",
"visit", "would", "vital", "voice"};
    secretWord = wordList[generateRandomIndex(wordList.size())];
}//end WordleGame()

//Input:The size of wordlist PW
// Process: Seed random number generator,generate a random index PW
//Output:Return a random index PW
int WordleGame::generateRandomIndex(int maxIndex) {
    return rand() % maxIndex;
}//end generateRandomIndex(int maxIndex)

//Input:A string of the user's input PW
// Process:Checks if the input is a 5-letter word and each character is alphabetical PW
//Output:A boolean indicating whether the input is valid (true) or not (false) PW
bool WordleGame::isValidInput(const string& input) {
    return input.length() == 5 && isalpha(input[0]) && isalpha(input[1]) &&
           isalpha(input[2]) && isalpha(input[3]) && isalpha(input[4]);
}// end isValidInput(const string& input)

//Input: strings of the secret word,the user's guess,a Guess struct to store the feedback PW
/*Process:Use a for loop to iterate through the first five characters of the strings.
          Checks if each character is in the related position,matches the corresponding character in secretWord.
          Append the corresponding feedback information ("Green," "Yellow," or "Gray") to result.feedback PW */
//Output: None PW
void WordleGame::provideFeedback(const string& secretWord, const string& guess, Guess& result) {
    result.feedback = "";
    for (int i = 0; i < 5; ++i) {    
        if (guess[i] == secretWord[i]) {
            result.feedback += "Green ";
        } else if (secretWord.find(guess[i]) != string::npos) {
            result.feedback += "Yellow ";
        } else {
            result.feedback += "Gray ";
        }//end if else
    }//end for
}//end provideFeedback(const string& secretWord, const string& guess, Guess& result)

// Input:Ask user to entry their guessing word PW
/*Process:Vlidate input,call function to get the feedback informtion,
          Use loop to allow user to attempt 6 times,
          a runtime_error is thrown if the input is invalid
          passing the secret word and the user's guess to compute and allocate memory to store feedback.PW*/  
/*Outcome:If the user successfully guesses the secret word, a congratulatory message is displayed.
          Using a pointer to dynamically allocate memory for Guess
          If the user exhausts all 6 attempts without a correct guess, a message is shown with the correct secret word and an encouragement to play again.
          In the case of a runtime error, an error message is displayed. PW*/
void WordleGame::playGame() {
    while (attempts < 6) {
        try {
            cout << "\nAttempt " << attempts + 1 << ": Enter a 5-letter word guess: ";
            string userGuess;
            cin >> userGuess;

            if (!isValidInput(userGuess)) {
                throw runtime_error("Invalid input. Please enter a valid 5-letter word.");
            }//end if

            Guess* guessResult = new Guess;
            provideFeedback(secretWord, userGuess, *guessResult);

            if (userGuess == secretWord) {
                cout << "Congratulations! You've guessed the secret word!" << endl;
                delete guessResult;            
                break;
            }//end if

            cout << "Result: " << guessResult->feedback << endl;
            if (attempts < 5) {
                cout << "Good progress! Keep it up!" << endl;
            }//end if
            attempts++;

            delete guessResult;         
            } //end try
            catch (const runtime_error& excpt) {
            cout << "Error: " << excpt.what() << endl;
            cin.clear();
            }//end catch
    }//end while

    if (attempts == 6) {
        cout << "Sorry, you've run out of attempts. The secret word was: " << secretWord << endl;
        cout << "Don't be discouraged! Play again!" << endl;
    }//end if
}//end playGame()

//Input: No direct input,invoke playgame() operation to ask user to entry guessing word PW
//Process:Display introduction of this game, instance wordgame class and invoke the method PW
//Output:Display introducition of this game. Use playgame()operation to print game details, feedback on each attempt, and additional encouragement messages PW
//The constructs I used: 1.if-else  2.loops  3. functions  4. vector  5.struct  6.OOP  7.pointers  8.stringstream  9.exceptions
int main() {
    WordleGame wordleGame;
    
    cout << "Welcome to the captivating realm of Wordle!"<< endl << endl;
    cout << "In this word-guessing game, your challenge is to unveil the secret word within a limit of 6 attempts. " << endl;
    cout << "Your task is to input a 5-letter word as your guess during each attempt." << endl << endl;
    cout << "The game employs a color-coded feedback system:" << endl << endl;
    cout << "Green: Correct letter in the correct position" << endl;
    cout << "Yellow: Correct letter but in the wrong position" << endl;
    cout << "Gray: Incorrect letter"<< endl << endl;
    cout << "Successfully deducing the secret word within the allotted attempts will lead to victory."<< endl;
    cout << "The game concludes after 6 attempts." << endl;
    cout<<"Are you ready to embark on this linguistic adventure? "<<endl;
    cout<<"Good luck!"<<endl;
   
    wordleGame.playGame();

    return 0;
}//end main()