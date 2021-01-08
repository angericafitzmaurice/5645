def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()


def create_list_from_file(filename):
    """Read words from a file and convert them to a list.
   
    Input:
    - filename: The name of a file containing one word per line.
    
    Returns
    - wordlist: a list containing all the words in the file, as strings.

    """
    wordlist = []
    with open(filename) as f:
        line = f.readline()
        while line:
            wordlist.append(line.strip())
            line = f.readline()
        return wordlist     


def strip_non_alpha(alpha):
    """ Remove non-alphabetic characters from the beginning and end of a string.

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle
    of the string should not be removed. E.g. "haven't" should remain unaltered."""
    if not alpha:
        return alpha
    for beg_str, element in enumerate(alpha):
        if element.isalpha():
            break
    for end_str, element in enumerate(alpha[::-1]):
        if element.isalpha():
            break
    return alpha[beg_str:len(alpha) - end_str]


def is_inflection_of(s1,s2):
    """ Tests if s1 is a common inflection of s2. 

    The function first (a) converts both strings to lowercase and (b) strips
    non-alphabetic characters from the beginning and end of each string. 
    Then, it returns True if the two resulting two strings are equal, or 
    the first string can be produced from the second by adding the following
    endings:
    (a) 's
    (b) s
    (c) es
    (d) ing
    (e) ed
    (f) d
    """
    # converts both strings to lowercase
    s1 = to_lower_case(s1)
    s2 = to_lower_case(s2)

    # calling strip_non_alpha() to strips non-alpha
    s1 = strip_non_alpha(s1)
    s2 = strip_non_alpha(s2)

    # calling same() to check if its an inflection of the other
    if same(s1, s2):
        return True
    else:
        return False

def same(s1,s2):
    "Return True if one of the input strings is the inflection of the other."

    # Get the len of the strings
    string1 = len(s1)
    string2 = len(s2)

    # initialize indexes
    s1_index = 0
    s2_index = 0

    while s1_index < string1 and s2_index < string2:
        if s1[s1_index] == s2[s2_index]:
            s1_index = s1_index + 1
        s2_index = s2_index + 1

    return s1_index == s2_index

def find_match(word,word_list):
    """Given a word, find a string in a list that is "the same" as this word.

    Input:
    - word: a string
    - word_list: a list of stings

    Return value:
    - A string in word_list that is "the same" as word, None otherwise.
    
    The string word is 'the same' as some string x in word_list, if word is the inflection of x,
    ignoring cases and leading or trailing non-alphabetic characters.
    """
    if word in word_list:
        filtered_word = list(filter(lambda w: word in w, word_list))
        print(''.join(filtered_word))
    else:
        return None

if __name__=="__main__":

    # Test strip_non_alpha
    # words = [",1what?1", "$av3", "haven't", "wh3r3s", "%hello", "hi!!", "p@r@ll3l", " ", "test" ]
    # output = [strip_non_alpha(word) for word in words]
    # print("Input: ", *words, sep=" ")
    # print("Output: ", *output, sep=" ")


    # Test is_inflection_of and same
    # first_string = input("first string: ")
    # second_string = input("second string: ")
    #
    # if is_inflection_of(first_string, second_string):
    #     print("Is Inflection: YES")
    # else:
    #     print("Is Inflection: NO")

    # Test find_match
    search = strip_non_alpha('12so!')
    list_of_strings = ['coding', 'is', 'fun', 'and', 'so', 'keep','learning']
    print(f'searching the word "{search}" in this list {list_of_strings}')

    for word in list_of_strings:
        strip_non_alpha(word)
        if is_inflection_of(search,word):
            find_match(search, list_of_strings)
            break











