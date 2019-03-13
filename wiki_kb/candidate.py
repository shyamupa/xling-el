__author__ = 'Shyam'


class Candidate:
    def __init__(self, surface, en_title, fr_title, is_gold, p_t_given_s, lang, src):
        self.surface = surface
        self.en_title = en_title
        self.fr_title = fr_title
        self.is_gold = is_gold
        self.p_t_given_s = p_t_given_s
        self.p_s_given_t = 0.0
        self.lang = lang
        self.src = src
        self.inv_edit_dist = None

    def __str__(self):
        return ";".join([self.surface,self.en_title,str(self.is_gold),str(self.p_t_given_s)])
