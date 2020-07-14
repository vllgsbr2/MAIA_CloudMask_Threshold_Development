import configparser

config = configparser.ConfigParser()
config.read('directories_config.txt')

print(config.sections())
