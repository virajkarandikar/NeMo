import numpy as np
import ctc_segmentation as cs
from nemo.collections import asr as nemo_asr


def prepare_tokenized_text_nemo(text, vocabulary):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_256")
    tokenizer = asr_model.tokenizer
    space_idx = vocabulary.index("▁")
    blank_idx = len(vocabulary) - 1

    ground_truth_mat = [[-1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        uttr_ids = []
        for id in token_ids:
            uttr_ids.append([id])
            # uttr_ids.append([space_idx])
        ground_truth_mat += uttr_ids #[[t] for t in token_ids]
        ground_truth_mat += [[space_idx]]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[space_idx]]

    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices


def prepare_tokenized_text(text, vocabulary):
    """Prepare the given tokenized text for CTC segmentation.

    :param config: an instance of CtcSegmentationParameters
    :param text: string with tokens separated by spaces
    :return: label matrix, character index matrix
    """
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.index_duration = 0.0799983368347339
    config.blank = len(vocabulary) -1
    config.space = "▁"
    # config.replace_spaces_with_blanks=True
    # config.tokenized_meta_symbol = "▁"

    ground_truth = [config.start_of_ground_truth]
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth[-1] == config.space:
            ground_truth += [config.space]
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add tokens of utterance
        for token in utt.split():
            if token in config.char_list:
                # import pdb; pdb.set_trace()
                if config.replace_spaces_with_blanks and not token.startswith(
                    config.tokenized_meta_symbol
                ):
                    ground_truth += [config.space]
                ground_truth += [token]
    # Add space to the end
    if not ground_truth[-1] == config.space:
        ground_truth += [config.space]
    print(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = 1
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] == config.space:
            ground_truth_mat[i, 0] = config.blank
        else:
            char_index = config.char_list.index(ground_truth[i])
            ground_truth_mat[i, 0] = char_index
    return ground_truth_mat, utt_begin_indices

# text = ["cat"]
# char_list = ["•", "UNK", "a", "c", "t", "cat"]
# # token_list = [cs.tokenize(utt) for utt in text]
# # print('token_list:', token_list)
# config = cs.CtcSegmentationParameters()
# ground_truth_mat, utt_begin_indices = cs.prepare_token_list(config, text)
# print(ground_truth_mat)
# array([[-1],
#        [ 0],
#        [ 5],
#        [ 0]])

# get_matricies("this is test")
# prepare_tokenized_text_nemo(["this is test"])
print()


def get_words(fpath, tokenizer):
    words = []
    with open(fpath, 'r') as f:
        for line in f:
            words.extend(line.strip().lower().split())
    # import pdb; pdb.set_trace()
    words = [" ".join(tokenizer.text_to_tokens(w)) for w in words]
    words = [w.replace("▁","") for w in words]
    print(words[:5])
    with open("/home/ebakhturina/data/segmentation/test/processed/1.txt", "w") as f_out:
        f_out.write("\n".join(words))
    return words


def prepare_text(config, text, char_list=None):
    """Prepare the given text for CTC segmentation.

    Creates a matrix of character symbols to represent the given text,
    then creates list of char indices depending on the models char list.

    :param config: an instance of CtcSegmentationParameters
    :param text: iterable of utterance transcriptions
    :param char_list: a set or list that includes all characters/symbols,
                        characters not included in this list are ignored
    :return: label matrix, character index matrix
    """
    # temporary compatibility fix for previous espnet versions
    if type(config.blank) == str:
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            if char.isspace() and config.replace_spaces_with_blanks:
                if not ground_truth.endswith(config.space):
                    ground_truth += config.space
            elif char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    print(f"ground_truth: {ground_truth}")
    # import pdb; pdb.set_trace()
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = max([len(c) for c in config.char_list])
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            span = span.replace(config.space, blank)
            if span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    return ground_truth_mat, utt_begin_indices


if __name__ == "__main__":
    # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_256")
    # vocabulary = asr_model.cfg.decoder.vocabulary + ["ε"]
    # tokenizer = asr_model.tokenizer
    #
    # text = "this is test"
    # import pdb; pdb.set_trace()
    # ground_truth_mat, utt_begin_indices = prepare_tokenized_text(text, vocabulary)


    from ctc_segmentation import CtcSegmentationParameters
    from ctc_segmentation import prepare_text

    char_list = ["<unk>", "'", "a", "ab", "able", "ace", "ach", "ack",
                 "act", "ad", "ag", "age", "ain", "al", "alk", "all", "ally", "am",
                 "ame", "an", "and", "ans", "ant", "ap", "ar", "ard", "are",
                 "art", "as", "ase", "ass", "ast", "at", "ate", "ated", "ater", "ation",
                 "ations", "ause", "ay", "b", "ber", "ble", "c", "ce", "cent",
                 "ces", "ch", "ci", "ck", "co", "ct", "d", "de", "du", "e", "ear",
                 "ect", "ed", "een", "el", "ell", "em", "en", "ence", "ens",
                 "ent", "enty", "ep", "er", "ere", "ers", "es", "ess", "est", "et", "f",
                 "fe", "ff", "g", "ge", "gh", "ght", "h", "her", "hing", "ht",
                 "i", "ia", "ial", "ib", "ic", "ical", "ice", "ich", "ict", "id", "ide",
                 "ie", "ies", "if", "iff", "ig", "ight", "ign", "il", "ild", "ill", "im",
                 "in", "ind", "ine", "ing", "ink", "int", "ion", "ions", "ip",
                 "ir", "ire", "is", "ish", "ist", "it", "ite", "ith", "itt", "ittle",
                 "ity", "iv", "ive", "ix", "iz", "j", "k", "ke", "king", "l",
                 "ld", "le", "ll", "ly", "m", "ment", "ms", "n", "nd", "nder", "nt", "o",
                 "od", "ody", "og", "ol", "olog", "om", "ome", "on",
                 "one", "ong", "oo", "ood", "ook", "op", "or", "ore", "orm", "ort",
                 "ory", "os", "ose", "ot", "other", "ou", "ould", "ound",
                 "ount", "our", "ous", "ousand", "out", "ow", "own", "p", "ph", "ple",
                 "pp", "pt", "q", "qu", "r", "ra", "rain", "re", "reat", "red",
                 "ree", "res", "ro", "rou", "rough", "round", "ru", "ry", "s", "se",
                 "sel", "so", "st", "t", "ter", "th", "ther", "ty", "u", "ually",
                 "ud", "ue", "ul", "ult", "um", "un", "und", "ur", "ure", "us", "use",
                 "ust", "ut", "v", "ve", "vel", "ven", "ver", "very", "ves", "ving", "w",
                 "way", "x", "y", "z", "ăť", "ō", "▁", "▁a", "▁ab",
                 "▁about", "▁ac", "▁act", "▁actually", "▁ad", "▁af", "▁ag", "▁al",
                 "▁all", "▁also", "▁am", "▁an", "▁and", "▁any", "▁ar",
                 "▁are", "▁around", "▁as", "▁at", "▁b", "▁back", "▁be", "▁bec",
                 "▁because", "▁been", "▁being", "▁bet", "▁bl", "▁br", "▁bu",
                 "▁but", "▁by", "▁c", "▁call", "▁can", "▁ch", "▁chan", "▁cl", "▁co",
                 "▁com", "▁comm", "▁comp", "▁con", "▁cont", "▁could", "▁d",
                 "▁day", "▁de", "▁des", "▁did", "▁diff", "▁differe", "▁different",
                 "▁dis", "▁do", "▁does", "▁don", "▁down", "▁e", "▁en",
                 "▁even", "▁every", "▁ex", "▁exp", "▁f", "▁fe", "▁fir", "▁first",
                 "▁five", "▁for", "▁fr", "▁from", "▁g", "▁get", "▁go",
                 "▁going", "▁good", "▁got", "▁h", "▁ha", "▁had", "▁happ", "▁has",
                 "▁have", "▁he", "▁her", "▁here", "▁his", "▁how", "▁hum",
                 "▁hundred", "▁i", "▁ide", "▁if", "▁im", "▁imp", "▁in", "▁ind", "▁int",
                 "▁inter", "▁into", "▁is", "▁it", "▁j", "▁just", "▁k", "▁kind",
                 "▁kn", "▁know", "▁l", "▁le", "▁let", "▁li", "▁life", "▁like",
                 "▁little", "▁lo", "▁look", "▁lot", "▁m",
                 "▁ma", "▁make", "▁man", "▁many", "▁may", "▁me", "▁mo", "▁more",
                 "▁most", "▁mu", "▁much", "▁my", "▁n", "▁ne", "▁need", "▁new",
                 "▁no", "▁not", "▁now", "▁o", "▁of", "▁on", "▁one", "▁only", "▁or",
                 "▁other", "▁our", "▁out", "▁over", "▁p", "▁part", "▁pe",
                 "▁peop", "▁people", "▁per", "▁ph", "▁pl", "▁po", "▁pr", "▁pre", "▁pro",
                 "▁put", "▁qu", "▁r", "▁re", "▁real", "▁really", "▁res",
                 "▁right", "▁ro", "▁s", "▁sa", "▁said", "▁say", "▁sc", "▁se", "▁see",
                 "▁sh", "▁she", "▁show", "▁so", "▁som", "▁some", "▁somet", "▁something",
                 "▁sp", "▁spe", "▁st", "▁start", "▁su", "▁sy", "▁t", "▁ta",
                 "▁take", "▁talk", "▁te", "▁th", "▁than", "▁that", "▁the", "▁their",
                 "▁them", "▁then", "▁there", "▁these", "▁they", "▁thing",
                 "▁things", "▁think", "▁this", "▁those", "▁thousand", "▁three",
                 "▁through", "▁tim", "▁time", "▁to", "▁tr", "▁tw", "▁two", "▁u",
                 "▁un", "▁under", "▁up", "▁us", "▁v", "▁very", "▁w", "▁want", "▁was",
                 "▁way", "▁we", "▁well", "▁were", "▁wh", "▁what", "▁when",
                 "▁where", "▁which", "▁who", "▁why", "▁will", "▁with", "▁wor", "▁work",
                 "▁world", "▁would", "▁y", "▁year", "▁years", "▁you", "▁your"]

    text = ["i ▁really ▁like ▁technology"]

    # text = ["I really like technology"]

    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_256")
    tokenizer = asr_model.tokenizer
    vocabulary = asr_model.cfg.decoder.vocabulary + ["ε"]
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.blank = len(vocabulary) - 1
    config.space = "▁"

    words = get_words("/home/ebakhturina/data/segmentation/test/data/text/1.txt", tokenizer)
    text  = words
    # text = ['i ▁ really ▁ like ▁ t e ch no lo g y ▁ ']
    # text = ['i', 'really', 'like','t e ch no lo g y']
    # import pdb; pdb.set_trace()
    for i in range(len(vocabulary)):
        if vocabulary[i].startswith("##"):
            vocabulary[i] = vocabulary[i].replace("##", "▁")

    ground_truth_mat, utt_begin_indices = prepare_tokenized_text(text, vocabulary)

    # #
    # #
    # # for i, v in enumerate(vocabulary):
    # #     if v.startswith("##"):
    # #         vocabulary[i] = vocabulary[i].replace("##", "▁")
    # # tokenizer = asr_model.tokenizer
    # # config.space = "▁"
    # # config.blank = -1
    # # config.char_list = vocabulary
    # # ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)
    #
    # text = ['this is sample', "test is here"]
    # ground_truth_mat, utt_begin_indices = prepare_tokenized_text_nemo(text, vocabulary)

    chars = []
    for row in ground_truth_mat:
        chars.append([])
        for ch_id in row:
            if ch_id != -1:
                chars[-1].append(vocabulary[int(ch_id)])

    print([print(x) for x in chars])
    import pdb;

    pdb.set_trace()


"""
[]
['<unk>']
['▁']
['r', '▁r']
['e', 're', '▁re']
['a']
['l', 'al', '▁real']
['l', 'll', 'all']
['y', 'ly', 'ally', '▁really']
['▁']
['l', '▁l']
['i', '▁li']
['k']
['e', 'ke', '▁like']
['▁']
['t', '▁t']
['e', '▁te']
['c']
['h', 'ch']
['n']
['o']
['l', 'ol']
['o']
['g', 'og', 'olog']
['y']
['<unk>']
['h']
['e']
['▁']
['q']
['u', 'qu', '▁qu']



"""