import os
import sys
import argparse
import numpy

pos_tags = [
    "AJ0",
    "AJC",
    "AJS",
    "AT0",
    "AV0",
    "AVP",
    "AVQ",
    "CJC",
    "CJS",
    "CJT",
    "CRD",
    "DPS",
    "DT0",
    "DTQ",
    "EX0",
    "ITJ",
    "NN0",
    "NN1",
    "NN2",
    "NP0",
    "ORD",
    "PNI",
    "PNP",
    "PNQ",
    "PNX",
    "POS",
    "PRF",
    "PRP",
    "PUL",
    "PUN",
    "PUQ",
    "PUR",
    "TO0",
    "UNC",
    "VBB",
    "VBD",
    "VBG",
    "VBI",
    "VBN",
    "VBZ",
    "VDB",
    "VDD",
    "VDG",
    "VDI",
    "VDN",
    "VDZ",
    "VHB",
    "VHD",
    "VHG",
    "VHI",
    "VHN",
    "VHZ",
    "VM0",
    "VVB",
    "VVD",
    "VVG",
    "VVI",
    "VVN",
    "VVZ",
    "XX0",
    "ZZ0",
    "AJ0-AV0",
    "AJ0-VVN",
    "AJ0-VVD",
    "AJ0-NN1",
    "AJ0-VVG",
    "AVP-PRP",
    "AVQ-CJS",
    "CJS-PRP",
    "CJT-DT0",
    "CRD-PNI",
    "NN1-NP0",
    "NN1-VVB",
    "NN1-VVG",
    "NN2-VVZ",
    "VVD-VVN",
    "AV0-AJ0",
    "VVN-AJ0",
    "VVD-AJ0",
    "NN1-AJ0",
    "VVG-AJ0",
    "PRP-AVP",
    "CJS-AVQ",
    "PRP-CJS",
    "DT0-CJT",
    "PNI-CRD",
    "NP0-NN1",
    "VVB-NN1",
    "VVG-NN1",
    "VVZ-NN2",
    "VVN-VVD",
]

ambiguity_tags = ["AJ0-AV0",
                  "AJ0-VVN",
                  "AJ0-VVD",
                  "AJ0-NN1",
                  "AJ0-VVG",
                  "AVP-PRP",
                  "AVQ-CJS",
                  "CJS-PRP",
                  "CJT-DT0",
                  "CRD-PNI",
                  "NN1-NP0",
                  "NN1-VVB",
                  "NN1-VVG",
                  "NN2-VVZ",
                  "VVD-VVN",
                  "AV0-AJ0",
                  "VVN-AJ0",
                  "VVD-AJ0",
                  "NN1-AJ0",
                  "VVG-AJ0",
                  "PRP-AVP",
                  "CJS-AVQ",
                  "PRP-CJS",
                  "DT0-CJT",
                  "PNI-CRD",
                  "NP0-NN1",
                  "VVB-NN1",
                  "VVG-NN1",
                  "VVZ-NN2",
                  "VVN-VVD", ]

pun = ['.', '?', ',', ':', ';', '-', '!']
pul = ['(', '[']
pur = [')', ']']
puq = ['"']
to = ['to']
vbd = ['was', 'were']
vbg = ['being']
vbi = ['be']
vbn = ['been']
vdb = ['do']
vdd = ['did']
vdg = ['doing']
vdi = ['do']
vdn = ['done']
ZZ0 = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
       'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# 1. check whether filtering calc is correct


def split_sentences(filename):

    lst_of_sentences = []
    sentence = []
    opening_quotes = False
    matching_quote = False
    markers_of_end_of_sentence = ['.', '?', '!']
    training_file = open(filename, "r")


    total_words = 0

    for line in training_file:

        if line.strip() != '':
            word_to_speech_tag = line.split(":")
            tag = word_to_speech_tag[1].strip()


            total_words += 1

            if ': :' in line:

                word = line[:line.index(':') + 2].strip()
                sentence.append(line.strip())

            else:
                word_to_speech_tag = line.split(":")
                word = word_to_speech_tag[0].strip()
                sentence.append(line.strip())

            # opening quote is found
            if opening_quotes == False and word == '"':
                opening_quotes = True

            # a matching quote is found -> could be end of sentence
            elif opening_quotes and matching_quote != True and word == '"':
                matching_quote = True

            elif opening_quotes and matching_quote:

                if word == '"' or word[0].isupper():
                    if word != '"':
                        opening_quotes = False

                    matching_quote = False

                    last_word_added = sentence.pop()

                    lst_of_sentences.append(sentence)
                    sentence = []

                    sentence.append(last_word_added)



                    # else -> keep looking for end of sentence

            # we found end of sentence -> normal case (no quotes)
            elif opening_quotes == False and word in markers_of_end_of_sentence:
                lst_of_sentences.append(sentence)
                sentence = []

    if sentence:
        lst_of_sentences.append(sentence)

    training_file.close()

    return lst_of_sentences


def calculate_transitional_or_observational_prob(
        speech_tag_before_to_occurence):

    total = 0

    # number of times total that prev_tag has appeared, previously in all sentences (transition)
    # number of words asscoiated with this tag
    for speech_tag in speech_tag_before_to_occurence:
        total += speech_tag_before_to_occurence[speech_tag]

    # prob(curr_tag | prev_tag) (transition)
    # p(curr_word | curr_tag)
    for speech_tag in speech_tag_before_to_occurence:
        speech_tag_before_to_occurence[speech_tag] = \
            speech_tag_before_to_occurence[speech_tag] / total

    return speech_tag_before_to_occurence


def calculate_probabilities(total_num_line, speech_tags_to_info):

    for speech_tag in speech_tags_to_info.keys():
        # initial probability
        speech_tags_to_info[speech_tag][0] = speech_tags_to_info[speech_tag][
                                                 0] / total_num_line

        speech_tags_to_info[speech_tag][
            1] = calculate_transitional_or_observational_prob(
            speech_tags_to_info[speech_tag][1])

        speech_tags_to_info[speech_tag][
            2] = calculate_transitional_or_observational_prob(
            speech_tags_to_info[speech_tag][2])

    return speech_tags_to_info


def speech_tag_in_dict(speech_tags_to_info, word_before, speech_tag, tag_before,
                       word, first_word):

    if speech_tag not in speech_tags_to_info.keys():
        speech_tags_to_info[speech_tag] = [0, {}, {}]

    # this word is after a period -> intial count += 1

    if first_word:
        speech_tags_to_info[speech_tag][0] += 1

    # check whether this first sentence in training file -> then no tag before
    if tag_before != None:
        # increment counter to account for fact that curr_tag came after prev_tag info
        # tag not encountered before
        if speech_tag in speech_tags_to_info[tag_before][1].keys():
            speech_tags_to_info[tag_before][1][speech_tag] += 1

        else:
            speech_tags_to_info[tag_before][1][speech_tag] = 1

    # increment counter to account for fact that curr_word is assoicated with curr_tag
    if word not in speech_tags_to_info[speech_tag][2].keys():
        speech_tags_to_info[speech_tag][2][word] = 1

    else:
        speech_tags_to_info[speech_tag][2][word] += 1

    return speech_tags_to_info


def get_correct_tag_for_test(lst_of_sentences):

    speech_tag_correct = []  # remove after

    for sentence in lst_of_sentences:

        for i in range(len(sentence)):

            if sentence[i].strip() != '':
                if ': :' in sentence[i]:
                    speech_tag = sentence[i][sentence[i].index(':') + 3:].strip(
                        '\n').strip(' ')

                else:
                    word_to_speech_tag = sentence[i].split(":")
                    speech_tag = word_to_speech_tag[1].strip('\n').strip(' ')

                speech_tag_correct.append(speech_tag)

    return speech_tag_correct


# [time it appeared at begging of sentence, {tag_after_it: times it appreaed after this tag}, {word_with_tag: time this word was associated with this tag}
def create_prob_tables(lst_of_sentences):

    # print(lst_of_sentences)

    speech_tags_to_info = {}
    word_before = None
    tag_before = None
    total_num_lines = len(lst_of_sentences)
    lst_of_words_in_text = []
    speech_tag_correct = []  # remove after

    for sentence in lst_of_sentences:

        for i in range(len(sentence)):

            if ': :' in sentence[i]:
                word = sentence[i][:sentence[i].index(':') + 1].strip().lower()
                speech_tag = sentence[i][sentence[i].index(':') + 3:].strip(
                    '\n').strip(' ')

            else:
                # get word and speech tag from line
                word_to_speech_tag = sentence[i].split(":")

                word = word_to_speech_tag[0].strip().lower()
                speech_tag = word_to_speech_tag[1].strip('\n').strip(' ')

            speech_tag_correct.append(speech_tag)

            if word not in lst_of_words_in_text:
                lst_of_words_in_text.append(word)

            # check whether the line is of first word -> important for inital prob
            first_word = False
            if i == 0:
                first_word = True

            speech_tags_to_info = speech_tag_in_dict(speech_tags_to_info,
                                                     word_before,
                                                     speech_tag,
                                                     tag_before, word,
                                                     first_word)

            word_before = word
            tag_before = speech_tag

    speech_tags_to_info = calculate_probabilities(total_num_lines,
                                                  speech_tags_to_info)

    return speech_tags_to_info, lst_of_words_in_text, speech_tag_correct


# inital prob [prob of a tag appearing at beggining of a sentence]
# trasntion matrix -> P(curr_tag| prev_tag)
def separate_probabilities(speech_tags_to_info, lst_of_words_in_text):

    initial_prob = []

    for i in range(91):
        initial_prob.append(0)

    transition_matrix = numpy.zeros(
        (91, 91))

    # tags prob
    observation_prob = numpy.zeros((91, len(lst_of_words_in_text)))

    for tag in speech_tags_to_info.keys():

        index_of_tag = pos_tags.index(tag)

        # 0 to the inital prob of all tags abive to be 0
        # if a tag does not appear in training file its intial prob is 0 -> we assigned
        initial_prob[index_of_tag] = speech_tags_to_info[tag][0]

        if initial_prob[index_of_tag] == 0:
            initial_prob[index_of_tag] = 0.1 * (10 ** (-100))

        else:
            initial_prob[index_of_tag] = initial_prob[index_of_tag] + 0.1 * (
                        10 ** (-100))

        # each row will be for tag and col will be the tags that were after it
        # used to calculate P(curr_tag| prev_tag)

        for next_tag in speech_tags_to_info[tag][1].keys():

            index_of_next_tag = pos_tags.index(next_tag)

            transition_matrix[index_of_tag, index_of_next_tag] = \
                speech_tags_to_info[tag][1][next_tag]

            if transition_matrix[index_of_tag, index_of_next_tag] == 0:
                transition_matrix[index_of_tag, index_of_next_tag] = 0.1 * (
                            10 ** (-100))

            else:
                transition_matrix[index_of_tag, index_of_next_tag] = \
                transition_matrix[index_of_tag, index_of_next_tag] + 0.1 * (
                        10 ** (-100))

        # each row will be for a tag and coloumns will be wards associated with that
        for i in range(len(lst_of_words_in_text)):

            word = lst_of_words_in_text[i]

            if word in speech_tags_to_info[tag][2].keys():
                observation_prob[index_of_tag, i] = \
                    speech_tags_to_info[tag][2][word]
            else:
                observation_prob[index_of_tag, i] = 0

            if observation_prob[index_of_tag, i] == 0:
                observation_prob[index_of_tag, i] = 0.1 * (10 ** (-100))

            else:
                observation_prob[index_of_tag, i] = observation_prob[
                                                        index_of_tag, i] + 0.1 * (
                                                            10 ** (-100))

    return initial_prob, transition_matrix, observation_prob


def words_in_test_file(test_file):

    E = []
    E_not_repeated = []
    orginal_E = []

    test_file_to_loop = open(test_file, "r")

    for line in test_file_to_loop:
        orginal_E.append(line.strip())
        word = line.strip().lower()
        E.append(word)

        if word not in E_not_repeated:
            E_not_repeated.append(word)

    return E, E_not_repeated, orginal_E


def possible_tags_for_curr_word(E, words_in_training, word_to_tag, t, prob,
                                curr_tag_to_after_tag, S):

    possible_tags_for_word = easy_tags(E, t, word_to_tag)

    if possible_tags_for_word == []:

        if E[t] in words_in_training:
            possible_tags_for_word = word_to_tag[E[t]]

        else:
            most_likley_tag_of_prev = numpy.argmax(prob[t - 1])
            possible_tags_for_word = curr_tag_to_after_tag[
                S[most_likley_tag_of_prev]]

    return possible_tags_for_word


def find_most_likely_prev_tag(tags_for_prev_word, prob, S, T, i, t, M,
                              E_not_repeated, E):

    # probabilities of curr_tag to prev_tag
    largest_prob_curr_tag_to_prev_tag = -1
    filtering_sum = 0

    # most likely previous tag
    x = None

    # if we consider all tags that come before tag i
    # and we consider all tags associated with the word t - 1
    # why would it ever be None?
    # it would only be none if the tag i is wrong for E_t

    for possible_prev_tag in tags_for_prev_word:

        # index of possible possible prev tag for E_t-1 in list of all tags (S)
        j = S.index(possible_prev_tag)

        # P(S_{t-1}|E_{t - 1})
        initial_prob = prob[t - 1, j]

        # need to write comments for what u did aboev

        # P(S_{t} | S_{t-1})
        # each row will be for P(curr_tag| prev_tag)
        # ex. P(AVO| tags that were previous to AV0)

        transition_prob = T[j, i]

        # each row will be for a tag and coloumn will be words associated with that
        observational_prob = M[i, E_not_repeated.index(E[t])]

        filtering_sum = filtering_sum + (initial_prob * transition_prob)

        prob_curr_tag_to_prev_tag = initial_prob * transition_prob * observational_prob
        # want to find the most likley previous_tag given the curr_tag is i

        if prob_curr_tag_to_prev_tag > largest_prob_curr_tag_to_prev_tag:
            x = j
            largest_prob_curr_tag_to_prev_tag = prob_curr_tag_to_prev_tag

    return x, filtering_sum


def calculate_prob_of_i_being_tag(prob, x, T, i, M, E_not_repeated, E, t):

    # P(S_{t-1} = x|E_{t - 1})
    initial_prob_ = prob[t - 1, x] + 0.1 * (10 ** (-100))



    # P(S_{t} = i | S_{t-1} = x)
    # each row will be for P(curr_tag| prev_tag)
    # ex. P(AVO| tags that were previous to AV0)

    # cannot be 0 as j is tag that comes before i
    transition_prob_ = T[x, i]

    # each row will be for a tag and coloumn will be words associated
    # with that

    # cannot be 0 as i is tags that were associated with E[t]
    observational_prob_ = M[i, E_not_repeated.index(E[t])]



    return initial_prob_ * transition_prob_ * observational_prob_


# E - list word in the test file
# S - list of all tags
# need to account for words that are in test file  but not in training file !!!!
def viterbi(E, S, I, T, M, word_to_tag, curr_tag_to_after_tag,
            words_in_training, E_not_repeated):

    # each row is for a observed word and each col is for a tag.
    # The prob rep the probability that a tag is associated with the observed word
    # each row is for a word and each column is for a tag
    prob = numpy.zeros((len(E), len(S)))
    prev = numpy.zeros(
        (len(E), len(S)))  # keeps track of bolded arrow for backtracking

    # determine values for t = 0
    # -> prob of what tag is associated with the first word-> P(S_0)P(E_0|S_0)

    for i in range(len(S)):

        prob[0, i] = I[i] * M[i, 0]
        prev[0, i] = None  # first time step so no arrows before it

    # loop through all the observed words except E_0
    for t in range(1, len(E)):

        if E[t] != "":
            # loop through all tags that were asssociated with this word in the trainig file
            # to determine a possible tag for E_t
            possible_tags_for_word = possible_tags_for_curr_word(E,
                                                                 words_in_training,
                                                                 word_to_tag, t,
                                                                 prob,
                                                                 curr_tag_to_after_tag,
                                                                 S)

            for possible_curr_tag in possible_tags_for_word:

                # index of possible curr tag for E_t in list of all tags (S)
                i = S.index(possible_curr_tag)

                # loop through all the tags that could have been prev_tag for curr_tag

                tags_for_prev_word = possible_tags_for_curr_word(E,
                                                                 words_in_training,
                                                                 word_to_tag,
                                                                 t - 1, prob,
                                                                 curr_tag_to_after_tag,
                                                                 S)
                # find the most likley tag for previous word
                x, filtering_sum = find_most_likely_prev_tag(tags_for_prev_word,
                                                             prob, S, T, i, t,
                                                             M, E_not_repeated,
                                                             E)

                # prob of the tag i being associated with E_t
                viterbi_prob = calculate_prob_of_i_being_tag(prob, x, T, i, M,
                                                             E_not_repeated, E,
                                                             t)

                prob[t, i] = viterbi_prob + (
                            M[i, E_not_repeated.index(E[t])] * filtering_sum)

                # the most likely prev_tag is S[x]
                prev[t, i] = x

    return prob, prev


def easy_tags(E, t, word_to_tag):

    if E[t] in word_to_tag.keys():
        if len(word_to_tag[E[t]]) == 1:
            return word_to_tag[E[t]]

    if E[t] in pun:
        possible_tags_for_word = ['PUN']

    elif E[t] in pul:
        possible_tags_for_word = ['PUL']

    elif E[t] in pur:
        possible_tags_for_word = ['PUR']

    elif E[t] in puq:
        possible_tags_for_word = ['PUQ']

    elif E[t] in ZZ0:
        possible_tags_for_word = ['ZZ0']

    elif E[t] == 'being':
        possible_tags_for_word = ['VBG']

    elif E[t] == 'having':
        possible_tags_for_word = ['VHG']


    elif E[t] in vdn:
        possible_tags_for_word = ['VDN']


    elif E[t] in vdg:
        possible_tags_for_word = ['VDG']

    elif E[t] in vdd:
        possible_tags_for_word = ['VDD']


    elif E[t] in vbn:
        possible_tags_for_word = ['VBN']

    elif E[t] in vbg:
        possible_tags_for_word = ['VBG']

    elif E[t] in puq:
        possible_tags_for_word = ['VBD']





    else:
        possible_tags_for_word = []

    return possible_tags_for_word






def backtrack(prob, prev, S, E, word_to_tag):

    sequence_of_tags = []
    t = len(E) - 1

    # find most likely tag for last word and add it to sequence
    # most_likley_tag_for_E_t, prev_tag = find_most_likely_tag_for_last_word(S, E, prob)
    most_likley_tag_for_E_t = S[numpy.argmax(prob[t])]
    prev_tag = numpy.argmax(prob[t])
    sequence_of_tags.append(most_likley_tag_for_E_t)

    # keep going till no bolded arrows
    # numpy.isnan(prev[t, prev_tag])
    while t > 0:
        # represent bolded arrow from E_t with tag prev_tag
        # given the word after has tag prev_tag
        #  prev_tag_index is the index of the tag to the word before it

        prev_tag_index = int(prev[t, prev_tag])

        possible_tag = easy_tags(E, t - 1, word_to_tag)

        if possible_tag != []:
            sequence_of_tags.append(possible_tag[0])
            prev_tag = S.index(possible_tag[0])

        else:


            sequence_of_tags.append(S[prev_tag_index])
            prev_tag = prev_tag_index

        t -= 1

    sequence_of_tags.reverse()

    return sequence_of_tags


# returns two dicts: one that maps words to the tags they were associated to
# in the training file (if any) the other maps tags to the tag previously to it
def organize_training_file_info(E, speech_tags_to_info):

    word_to_tag = {}
    curr_tag_to_prev_tag = {}
    curr_tag_to_after_tag = {}

    # creates a dict with keys being words from the test file
    for i in range(len(E)):
        word_to_tag[E[i]] = []

    for tag in pos_tags:
        curr_tag_to_prev_tag[tag] = []
        curr_tag_to_after_tag[tag] = []

    for tag in speech_tags_to_info:

        # creates a dict mapping words to
        # tags they were associated in training file

        for word in speech_tags_to_info[tag][2].keys():
            if word in word_to_tag.keys():
                word_to_tag[word].append(tag)

        # creates a dict mapping tags to the tags that appear before it
        for tag_after in speech_tags_to_info[tag][1].keys():
            curr_tag_to_prev_tag[tag_after].append(tag)
            curr_tag_to_after_tag[tag].append(tag_after)

    return word_to_tag, curr_tag_to_prev_tag, curr_tag_to_after_tag





# need to consider the case of multiple training files !!!!
def find_sequence_of_speech_tags(lst_of_sentences, lst_words_in_test,
                                 lst_words_in_test_not_repeated, orginal_E):

    # create probability table given training file
    # speech_tag_correct -> for test will delete after
    # word in training file can also possibly remove
    speech_tags_to_info, words_in_training_file, speech_tag_correct2 = create_prob_tables(
        lst_of_sentences)

    E = lst_words_in_test

    # we don't want repeated words so prob and info for a words kept all together
    word_to_tag, curr_tag_to_prev_tag, curr_tag_to_after_tag = organize_training_file_info(
        lst_words_in_test_not_repeated, speech_tags_to_info)

    # we don't want repeated words so prob for a words kept all together
    I, T, M = separate_probabilities(speech_tags_to_info,
                                     lst_words_in_test_not_repeated)

    # we need all words in E as we are going through it
    prob, prev = viterbi(E, pos_tags, I, T, M, word_to_tag,
                         curr_tag_to_after_tag, words_in_training_file,
                         lst_words_in_test_not_repeated)

    sequence_of_tags = backtrack(prob, prev, pos_tags, orginal_E, word_to_tag)



    return sequence_of_tags


def set_up(training_file_list, test_file, outputfile):


    lst_of_sentences = []

    for train_lst_name in training_file_list:
        lst_of_sentences += split_sentences(train_lst_name)

    lst_of_words_in_test_file, lst_of_words_in_test_file_not_repeated, orginal_E = words_in_test_file(
        test_file)

    sequence_of_tags = find_sequence_of_speech_tags(lst_of_sentences,
                                                    lst_of_words_in_test_file,
                                                    lst_of_words_in_test_file_not_repeated, orginal_E)

    f = open(outputfile, 'w')

    for i in range(len(orginal_E)):
        if orginal_E[i] == "":
            f.write(orginal_E[i] + "\n")
        else:
            f.write(orginal_E[i] + " : " + sequence_of_tags[i] + "\n")
    f.close()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]

    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))



    print("Starting the tagging process.")

    set_up(training_list, args.testfile, args.outputfile)








