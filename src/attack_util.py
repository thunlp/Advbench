import random




def insert_long_distractor(sentence,begsl,endsl,ch):
    # insert_pool = [str(i) for i in list(range(0,10))]  + ['!','&','%','$','#']
    # k = 10
    # sent_len = len(sentence.split(' '))
    # distractor = random.choice(insert_pool) * sent_len * k
    # distractor =" up" *8
    beg_word = ch+" "
    end_word = " "+ch
    for_distractor =beg_word *begsl
    # distractor =" reuters" *30
    distractor =end_word *endsl
    # distractor =" to" *180
    return for_distractor+sentence + distractor

# def insert_long_distractor(sentence):
#     # insert_pool = [str(i) for i in list(range(0,10))]  + ['!','&','%','$','#']
#     # k = 10
#     # sent_len = len(sentence.split(' '))
#     # distractor = random.choice(insert_pool) * sent_len * k
#     distractor =" up" *8
#     # beg_word = ch+" "
#     # end_word = " "+ch
#     # for_distractor =beg_word *begsl
#     # # distractor =" reuters" *30
#     # distractor =end_word *endsl
#     # distractor =" to" *180
#     return sentence + distractor

def insert_space(word):
    word_len = len(word)
    if word_len < 5:
        return None
    insert_pos = random.choice(list(range(0, word_len-1))) + 1  # word_len=5, 0,1,2,  ->  1,2,3   fooli   f ooli   fo oli foo li  fool i
    new_word = word[: insert_pos] + ' ' + word[insert_pos: ]
    return new_word

def insert_irrelevant(word):
    word_len = len(word)
    if word_len < 5:
        return None
    insert_pos = random.choice(
        list(range(0, word_len - 1))) + 1  # word_len=5, 0,1,2,  ->  1,2,3   fooli   f ooli   fo oli foo li  fool i
    # insert_pool = ['$', '#', '@', '%', '&', '!', '.','<', '^','>','*']
    # insert_pool = ['it']
    insert_pool = ['$', '#', '@', '%', '&', '!','*']
    insert_char = random.choice(insert_pool)
    new_word = word[: insert_pos] + insert_char + word[insert_pos:]
    return new_word

def delete_char(word):
    word_len = len(word)
    if word_len < 5:
        return None
    insert_pos = random.choice(list(range(0, word_len - 1))) + 1
    new_word = word[: insert_pos] + word[insert_pos+1:]
    return new_word



def swap_char(word):
    word_len = len(word)
    if word_len < 5:
        return None
    insert_pos = random.choice(list(range(0, word_len - 2))) + 1
    i = word[insert_pos]
    j = word[insert_pos+1]
    new_word = word[: insert_pos] + j + i + word[insert_pos+2:]
    return new_word






def sub_char(word):
    word_len = len(word)
    if word_len < 5:
        return None

    # sub_dict ={ '-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l', '0': 'O',
    #      "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ',
    #      'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս',
    #      'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}
    sub_dict = {
        'o': '0',
        'l': '1',
        'i': '1',
        's': '$',
        # 'a': ''
    }
    sub_list = ['o', 'l', 'i','s']
    # sub_list = ['-', '9', '8', '7', '6', '5', '4', '3', '2', '1', '0',
    #      "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    #      'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    #      'v', 'w', 'x', 'y', 'z']
    if len(set(sub_list) & set(word)) == 0:
        return None
    random.shuffle(sub_list)
    for target_char in sub_list:
        if target_char in word:
            new_word = word.replace(target_char, sub_dict[target_char])
            return new_word
        else:
            continue
    raise RuntimeError





def insert_period(sentence):
    word_li = sentence.split(' ')
    new_word_li = []
    for word in word_li:
        new_word_li.append(word)
        new_word_li.append('.')
    return ' '.join(new_word_li)





if __name__ == '__main__':
    '''test'''
    test_sentence = '5 of arthritis patients in Singapore take Bextra or Celebrex &lt;b&gt;...&lt;/b&gt; SINGAPORE : Doctors in the United States have warned that painkillers Bextra and Celebrex may be linked to major cardiovascular problems and should not be prescribed.'
    # print(insert_long_distractor(test_sentence))
    print(insert_period(test_sentence))
    word = 'foolish'
    print(insert_space(word))
    print(insert_irrelevant(word))
    print(delete_char(word))
    print(swap_char(word))
    print(sub_char(word))







