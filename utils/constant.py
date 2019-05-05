"""
Define constants
"""
PAD_ID = 0
UNK_ID = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'

VOCAB_PREFIX = [PAD_WORD, UNK_WORD]


START_TAG = "<START>"
STOP_TAG = "<STOP>"


NER_TO_ID = {'O': 0, 'location': 1, 'person': 2,'organization':3}

NER_TAG_TO_ID = {'O': 0, 'b-location': 1, 'i-location': 2, 'l-location': 3, 'u-location': 4,
                 'b-person': 5, 'i-person': 6, 'l-person': 7, 'u-person': 8,'b-organization': 9,
                 'i-organization': 10, 'l-organization': 11, 'u-organization':12,
                 START_TAG: 13, STOP_TAG: 14}


LABEL_TO_ID = {'None': 0, 'founders': 1, 'place_of_birth': 2, 'place_of_death': 3
               , 'major_shareholder_of': 4, 'people': 5, 'neighborhood_of': 6,
               'location': 7, 'industry': 8, 'place_founded': 9, 'country': 10, 'teams': 11,
               'nationality': 12, 'religion': 13, 'advisors': 14, 'ethnicity': 15,
               'geographic_distribution': 16, 'company': 17, 'major_shareholders': 18,
               'place_lived': 19, 'profession': 20, 'capital': 21, 'contains': 22,
               'administrative_divisions': 23, 'children': 24}



