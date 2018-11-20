import copy
import sys


def permutations(string, permutation_list, step=0):
    # if we've gotten to the end, print the permutation
    if step == len(string):
        permutation_list.append("".join(string))

    # everything to the right of step has not been swapped yet
    for i in range(step, len(string)):
        # copy the string (store as array)
        string_copy = [character for character in string]

        # swap the current index with the step
        string_copy[step], string_copy[i] = string_copy[i], string_copy[step]

        # recurse on the portion of the string that has not been swapped yet (now it's index will begin with step + 1)
        permutations(string_copy, permutation_list, step + 1)


def get_dictionary_words(word):
    word_perms = []
    dict_words = []
    permutations(word, word_perms)

    for perm in word_perms:
        if perm in word_list:
            dict_words.append(perm)

    return dict_words


def find_valid_words(letters):
    # print(letters)
    avail = [0 for i in range(26)]
    for c in letters:
        # print(c)
        index = ord(c) - ord('a')
        # print(index)
        avail[index] += 1
    result = []

    # only search words which are the same length as letters
    word_list_upper_bound=(len(letters) + 1)
    word_list_lower_bound=len(letters)
    if word_list_upper_bound > 28:
        print("index out of bound for the short list")
    if word_list_lower_bound < 3:
        return result


    short_list = word_list[word_list_idx[word_list_lower_bound]:word_list_idx[word_list_upper_bound]]


    for word in short_list:
        count = [0 for i in range(26)]
        ok = True
        for c in word:
            index = ord(c) - ord('a')
            count[index] = count[index] + 1
            if count[index] > avail[index]:
                ok = False;
                break;

        if ok and len(word) == len(letters):
            result.append(word)
            # print(word)

    return result


def perform_alterations(word):
    # Stores all next possible dictionary words
    next_words = []

    if len(word) >= 4:
    # Removing single letters
        for index, alphabet in enumerate(word):
            if alphabet not in goal_word or (goal_word.count(alphabet) < word.count(alphabet)):
                #count = check_string.count(char)
                # print(alphabet,index)
                temp_word = word
                new_word = temp_word[:index] + temp_word[index + 1:]
                dict_words = find_valid_words(new_word)
                if len(dict_words) != 0:
                    next_words.extend(dict_words)

    # Adding single letters
    for alphabet in alphabet_list:
        if alphabet in goal_word:
            new_word = word
            new_word = new_word + alphabet
            dict_words = find_valid_words(new_word)
            if len(dict_words) != 0:
                for dict_word in dict_words:
                    next_words.append(dict_word)

    return next_words

def write_ladder_to_file(output_ladder):
    output_file = open("output.txt", 'w+')
    for word in output_ladder:
        if word != starting_word[-1]:
            output_file.write('\n')
        output_file.write(word)
    output_file.close()
    return 0


def get_next_hop(input_ladder):
    next_hop = []
    last_word = input_ladder[-1]

    next_words = perform_alterations(last_word)

    for next_word in next_words:
        if next_word not in input_ladder:
            next_hop.append(copy.deepcopy(input_ladder))
            next_hop[-1].append(next_word)

    return next_hop


# ============== SCRIPT ============ #
word_list = []
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z']
# alphabet_list = ['a', 'b', 'c']

starting_word = [sys.argv[1], ""]
goal_word = sys.argv[2]

#starting_word = ["croissant", ""]
#goal_word = "baritone"

processing_queue = []
processed_ladders_queue = []
ladder_solution = []

processing_queue.append(starting_word)

word_list_file = open("wordList.txt", 'r')
word_list = word_list_file.read().splitlines()
word_list_file.close()
word_list.sort(key = lambda s: len(s))
word_list_idx = dict()

prev_len = 0
for idx, word in enumerate(word_list):
    if (len(word) > prev_len):
        prev_len = len(word)
        word_list_idx.update({prev_len: idx})

hop_num = 0
current_hop_word_list = []
while len(processing_queue) != 0:
    next_ladder = processing_queue.pop(0)

    if next_ladder[-1] == '':
        next_ladder.remove('')

    test_word = next_ladder[-1]
    if test_word == goal_word:
        #print("Found solution to ladder problem.")
        #print(next_ladder)
        write_ladder_to_file(next_ladder)
        break
    else:
        processed_ladders_queue.append(next_ladder)

    # If the queue is empty then we need to go to the next hop
    if len(processing_queue) == 0:

        #hop_num += 1
        #print("Calculating hop " + str(hop_num) + "...")

        for ladder in processed_ladders_queue:
            iteration_ladder = get_next_hop(processed_ladders_queue.pop(0))
            if iteration_ladder:
                iteration_ladder2 = []
                for iteration_index in iteration_ladder:
                    current_word = iteration_index[-1]

                    if current_word not in current_hop_word_list:
                        current_hop_word_list.append(current_word)
                        iteration_ladder2.append(iteration_index)

                processing_queue.extend(iteration_ladder2)
