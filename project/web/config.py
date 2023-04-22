class Config(object):
    def __init__(self):
        self.DEBUG = True
        self.SECRET_KEY = "k34mx2w1ps4qqlx95r"
        self.UPLOADS = "./uploads"
        self.CLIENT_IMAGES = "./static/client/img"
        self.SESSION_COOKIE_SECURE = True

class DevConfig(Config):
    pass