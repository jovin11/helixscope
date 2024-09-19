import os
import sys
import math
import random
import contextlib
import numpy as np
import tensorflow as tf
from Bio.SeqUtils import seq1
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, TimeDistributed, MaxPooling1D
from tensorflow.keras.layers import Input, Embedding, GlobalMaxPooling1D, Masking, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from Bio.PDB import PDBList, PDBParser, DSSP

# ------------------------------------------------------------------------------------------------------------------------------------
# Use the following code to download a random set of PDB files

# # Initialize the PDBList object
# pdb_list = PDBList()

# # Fetch the full list of all PDB IDs (this may take a moment)
# all_pdb_ids = pdb_list.get_all_entries()

# # Randomly select 500 PDB IDs from the full list
# selected_pdb_ids = random.sample(all_pdb_ids, 1000)

# # Print or store the selected PDB IDs
# print(selected_pdb_ids)

# ------------------------------------------------------------------------------------------------------------------------------------

# 634 -> 1179 -> 2097 ids
# List of more PDB IDs for a larger dataset (example IDs, replace with actual ones)
pdb_ids = ['1A8M', '1BNA', '1C0A', '1D3Z', '1E7P', '1F9T', '1GID', '1H9N', '1JFF', '1KX5', '1L2Y', '1N8O', '1O5R', '1P6X', '1R4Z', 
           '1S7Y', '1T8W', '1U9Z', '1A6M', '1BVC', '1CEX', '1D4T', '1E7O', '1F3H', '1GFL', '1HNI', '1IDH', '1J5E', '1K5D', '1M1T', 
           '1N9X', '1O3Y', '1Q8Z', '1R2Y', '1S1D', '1U6Q', '1V3W', '1W1Z', '1X3A', '1Y4B', '1Z6C', '2A7D', '2B8E', '2C9F', '2D0G', 
           '2E1H', '2F2I', '2G3J', '2H4K', '2I5L', '2J6M', '2K7N', '2M9P', '2N0Q', '2P2S', '2Q3T', '2R4U', '2V8Y', '2W9Z', '3A1A', 
           '3C3C', '3D4D', '3E5E', '3F6F', '3G7G', '3H8H', '3I9I', '3J0J', '3K1K', '3L2L', '3M3M', '3N4N', '3O5O', '3P6P', '3Q7Q', 
           '3R8R', '3S9S', '3T0T', '3U1U', '3V2V', '3W3W', '4B2B', '4C3C', '4D4D', '4E5E', '4F6F', '4G7G', '4H8H', '4I9I', '4J0J', 
           '4K1K', '4L2L', '4M3M', '4N4N', '4O5O', '4P6P', '4Q7Q', '4R8R', '4V2V', '4X4X', '4Y5Y', '4Z6Z', '5A1A', '5C3C', '5D4D', 
           '5E5E', '5F6F', '5H8H', '5I9I', '5J0J', '5K1K', '2HRK', '3A5X', '4XYZ', '6ZP7', '3R2C', '2FL7', '4H57', '1BCX', '2B4D', 
           '5C67', '5N8E', '5O8F', '5P8G', '5Q8H', '5R8I', '5S8J', '5T8K', '5U8L', '5V8M', '5Y8P', '5Z8Q', '6A8R', '6B8S', '6D8U', 
           '6E8V', '6F8W', '6G8X', '6H8Y', '6I8Z', '6J90', '6K91', '6L92', '6M93', '6N94', '6O95', '6P96', '6U9B', '6V9C', '6W9D', 
           '6Y9F', '6Z9G', '7B9I', '7D9K', '7E9L', '7G9N', '7K9R', '7L9S', '7M9T', '7N9U', '7P9W', '7R9Y', '7S9Z', '7T10', '7U11', 
           '7V12', '7W13', '7X14', '7Y15', '8A17', '8B18', '8C19', '8E1B', '8F1C', '8H1E', '8I1F', '8K1H', '8P1M', '8Q1N', '8R1O', 
           '8T1Q', '8U1R', '7BJB', '1HAI', '5LOM', '7KE6', '1NQ6', '6SVB', '4MUF', '8D5W', '4INB', '4UAW', '4IQI', '6X0O', '1OUK', 
           '7V3G', '6SST', '8BX6', '8F5M', '2OV4', '7FTT', '4LLH', '7MYA', '6O4G', '1PR5', '2DV7', '3VC0', '8EYI', '7KJ0', '7WF5', 
           '1V61', '3S2E', '1X46', '3PTM', '7SXQ', '1G4C', '6UJN', '4CTX', '4L3P', '7BY5', '6HK4', '8I9E', '1GE5', '1KGW', '6YTZ', 
           '3VR2', '6I1Z', '2LRA', '6RIT', '4KI1', '1M1Q', '5DFP', '2FQL', '1W5Z', '3KUT', '1ZZ3', '1IWZ', '8DFH', '2A87', '3SPZ', '3HNL', '7L7S', '6HVA', '7JGP', '2UVD', '2J3G', '3Q3I', '3V0N', '2EC5', '2O70', '3NYS', '7N7P', '6SVM', '6BRR', '5TQE', '2RS7', '1VLL', '2RT6', '2IED', '1QAJ', '4KMS', '1DJ0', '6OS0', '4Y9Y', '3GAC', '1Z6A', '7MVS', '1P1G', '7FBQ', '2GMS', '2RAT', '1R5B', '6ANU', '1RCJ', '4CJ0', '2UZS', '2RK9', '1PPJ', '3HWX', '6NQY', '6WL6', '1T1Y', '5EPS', '3PXN', '2DP4', '2K2D', '2OYK', '7AEG', '3LJ0', '4AKJ', '8FQH', '4TU7', '6M9S', '4HWB', '5B3K', '1MA7', '6DI6', '8Y6P', '4CT6', '2E09', '1M5X', '1CKF', '8UH9', '4M8Q', '3DCA', '2DVW', '6MBE', '8J78', '5K0R', '5JA3', '4OI1', '8FU4', '8G6Q', '3BH8', '5SD0', '6LN5', '4JDI', '6FU4', '4PI3', '7ZWF', '8IZY', '7DK6', '1D7L', '5YT2', '6INT', '4QSS', '8GFZ', '8C6H', '5K90', '4OFP', '5P6U', '6EOD', '4IQV', '6PTN', '2H16', '4I19', '6WQ9', '5HDH', '5JQX', '193L', '6GW2', '8D4T', '5XB0', '4XF2', '2K51', '3Q6K', '6U7P', '1LQA', '7RZ1', '5D3R', '6EIR', '8VTZ', '8TVZ', '2Q8L', '5T8U', '1N76', '3TGV', '3IY1', '1NP1', '4KP5', '7GU6', '2F9R', '6I04', '2JVG', '3I7H', '1MIY', '3ZS1', '3TKA', '7S0R', '7MK3', '2H4E', '3OQL', '6JTJ', '1ZOR', '6R53', '6IKO', '7P8X', '7B4S', '4PPJ', '8W0D', '8I1D', '1RQA', '4ZJG', '3RPU', '2GVG', '6VD2', '8J5W', '7YJK', '3UHN', '6BDU', '6RFB', '8SHK', '6R0H', '8E6J', '7NGY', '6Y75', '5W2U', '2VOV', '1AA2', '6WOG', '1VGG', '8P22', '3KDS', '5Y51', '1IZL', '8KE8', '1UW1', '2CNF', '4CNU', '1S8I', '2P1Q', '5BP1', '4PJ7', '5WYN', '6P3L', '4MB3', '6M5P', '3G21', '4OFH', '7D6H', '6HTI', '7URF', '1PH0', '6HYP', '8HUB', '3K31', '3A2S', '3BOI', '2OCA', '3PMV', '4RF2', '7K27', '7GH7', '7L85', '2J8T', '2NMY', '8WLS', '5Z9P', '1L9S', '5DWI', '2MUM', '3OGV', '1KSL', '1B0Y', '1J82', '7X4U', '4JYG', '1RUH', '6YB1', '5QAF', '5TEA', '2CCF', '3RGH', '3RMH', '5P7A', '6FE5', '8TYS', '6WF5', '4U58', '2L7U', '6HTA', '1WWF', '7UFA', '6O0A', '1CZ7', '6RKU', '7EEN', '5DF9', '1Z3A', '1BD4', '5C4Y', '2KRB', '2KJE', '6GCF', '7ASG', '6C33', '4LFB', '5A8F', '5BYJ', '6HK1', '1IWN', '1RSN', '6AAH', '1NMA', '3K5Y', '3GHD', '2Y5N', '7PB5', '3IEY', '8QRV', '2OMK', '7XUD', '6RHT', '2PS5', '2BCD', '4V1A', '6MJS', '4GLO', '4OMK', '7AM3', '2VQI', '2KDT', '5UIP', '5HBY', '1K32', '5AME', '2B7P', '2XEB', '5NK4', '1ZQH', '1NBE', '2YMP', '4O24', '5T9U', '5MMW', '2KF7', '6TJD', '5GR1', '7MHI', '3IP4', '5TX3', '3ZDH', '3A75', '4B89', '6KGT', '8UP8', '5INQ', '6HP0', '7L67', '8QAB', '2O5Y', '7KRB', '1WT8', '7F12', '8C5L', '3ZEZ', '5WLU', '8QAH', '6H6D', '1P2G', '3OJJ', '7AJS', '7UEQ', '3C5D', '4J49', '6N27', '8DV7', '5IYQ', '6BMK', '1QJA', '3P2J', '7PRY', '3CI5', '5UOI', '3I2S', '2HYI', '5FOJ', '1EEP', '8A7L', '1BRY', '8HA2', '1XL1', '1KDC', '1N8K', '3JT5', '5L80', '2EF7', '1F81', '4CUR', '5GIC', '2AMM', '5YDU', '3QZA', '5US3', '7AS6', '3SJ5', '5CR2', '2FYD', '5TBJ', '8PRE', '6DAJ', '6E18', '3GGV', '7DBI', '1NTP', '2V05', '3DSN', '5YDE', '7XO1', '5GIT', '5R4K', '3VCE', '3NES', '5K75', '2ZLW', '2XF0', '7LH8', '3W2O', '5QI3', '1AMI', '4NGI', '7YGD', '8TR2', '7YA7', '8CP0', '1DYT', '8Q10', '6DBW', '7C1E', '1I4Q', '4XTN', '2GIA', '4U9H', '2HFR', '2WUK', '4CK1', '2QTY', '2Q4T', '3ZIM', '6XFG', '3ERP', '5Q9U', '1P3K', '8I2G', '7DLY', '1FO7', '1NSR', '7O24', '8AI6', '3V97', '1LX7', '5DOM', '8IH0', '3CQ0', '5ZCA', '2CLR', '5O1F', '4MUN', '4LU6', '3SOW', '5PMK', '6BKF', '3GVD', '2DM1', '4GGL', '3SCY', '1SW2', '1C17', '3HJJ', '1EUR', '5EBX', '1BGV', '2PXW', '3UV0', '6KGE', '2OKK', '4DMU', '6MLM', '4S3J', '3RU5', '1WBV', '6W8L', '8T9N', '1W9C', '6OKJ', '8H8L', '7B1A', '8AKB', '1HRM', '1WC0', '7M75', '3PVF', '3M97', '3ZFF', '7MLK', '3C20', '6RQ1', '7MPW', '2B3P', '5KTT', '4JUO', '4EGN', '1UTU', '1R66', '6Q91', '4JAW', '4R6T', '1Y9E', '1OC6', '4AYX', '7VB6', '3VQ1', '3FU6', '3I4O', '2VKH', '5AC2', '2OBA', '1WMX', '4UGZ', '1P1N', '1P1M', '6FAU', '5C3U', '7AN5', '3ZMF', '7F84', '3QLG', '7CWB', '8Q9K', '2BTC', '4FP7', '3MRQ', '7E49', '6D6H', '2XX8', '5KXI', '8F6P', '8XNG', '5QDV', '8AES', '6IVR', '1HFO', '1EYV', '7LVG', '7BSO', '1V3A', '7AZ1', '5Y7Y', '3S7L', '2HVP', '5WDU', '4TKR', '4K96', '4AEI', '5FI2', '8BEI', '6H1F', '7CMO', '2YOG', '8HY1', '5APH', '2Q0X', '5HMA', '4WHA', '3T26', '8AX8', '8ERF', '8IVS', '7AUX', '1ZVT', '5JG1', '7AME', '1R3F', '8QV4', '3CIC', '2LFJ', '2FAH', '287D', '1AL5', '1IRU', '6N87', '6DZ5', '2ZN8', '7TQU', '7LZW', '2ODC', '1YKF', '1O9Y', '4DJE', '1WAX', '7PEJ', '2KB9', '4GXL', '1ORF', '5HYY', '3FSW', '1CLY', '3FA8', '6O77', '1HJK', '7U04', '7G0I', '7TB9', '4XOW', '5T2Y', '3ASY', '3M5M', '3S46', '1RUO', '4BR9', '3MAJ', '1J16', '6P17', '3OU0', '1K4A', '5PO3', '1DWC', '6OO0', '1EU5', '3OLV', '2ELU', '5THI', '1CRB', '2PHM', '6XYY', '3NZH', '5V8P', '1XXG', '7YEN', '7ZM6', '2IL1', '5JDX', '8FDS', '4RIR', '4C39', '4N16', '5AM6', '3CWU', '2R2O', '5YX1', '4XBU', '6K2T', '6SDT', '3ZQS', '2OIX', '3VHA', '3FLM', '3FW1', '6O6M', '4P0V', '3L0C', '1AA1', '2EQY', '4CFP', '1OL5', '5THT', '2WG6', '1A2T', '3BWB', '4HXL', '2GFT', '3W5C', '8G2L', '2V9B', '4JM8', '1S5S', '7X85', '8GVF', '6HK8', '1KZL', '2XGN', '6Z41', '1R30', '7R5F', '2IXV', '2WAE', '4X9W', '2NWL', '1RBP', '3T0S', '2BXK', '4OEX', '4KRD', '5C1B', '7U9P', '4BDX', '2PLV', '2HTD', '4IR7', '8ICQ', '6W9A', '4I3L', '4EJ0', '8D3A', '7R7T', '5KWY', '7JQR', '3OQT', '2O7H', '8BYJ', '4XT7', '4UEG', '1NN2', '5Y9B', '6Z8Y', '5PKB', '1OUC', '8J2J', '7XIX', '8EV5', '7CJL', '5M4X', '8THM', '1B22', '4N90', '1Q0Q', '4GSL', '4NU5', '1KT2', '7P5Y', '7COH', '7LM7', '7UY2', '1FGD', '1QD0', '1TY4', '6OPJ', '2VXM', '4NZ8', '6CWS', '8EBB', '1W33', '7PK1', '1YLE', '8H1W', '4XSZ', '1GZF', '8C31', '4REC', '7ONB', '1Y1B', '1GAY', '5VO5', '1ROF', '2C9W', '6L0V', '3RCW', '4HWV', '5Z9F', '3GBO', '2I15', '5C31', '3JSS', '6XK9', '1EHN', '3M9X', '5JNQ', '3FY3', '5J85', '4I3V', '4MQG', '5SSI', '6UIE', '1TBT', '1MN8', '6Z2F', '8XD3', '1XCU', '3OF6', '6MP8', '7E1A', '3EMR', '7QL1', '3EU4', '2LML', '5DJ0', '3ZKE', '3WW4', '6YE1', '8WR8', '7T55', '3JSU', '3QZU', '7S9A', '6DN2', '7SIL', '3CDH', '1UJ8', '2IEE', '8K8E', '7PY3', '6RWA', '4HUK', '6PGJ', '4K3A', '2N8X', '7SNX', '5SL6', '7QSP', '7TPD', '2BN6', '5W22', '1UBP', '5ERP', '1IUH', '1ICI', '1KM0', '7DTS', '4V3Q', '5F5S', '2N0B', '8G0Q', '6YYC', '4MSP', '1HJ3', '2ZSZ', '1OIS', '6SFC', '1O15', '6TBH', '5JSA', '3VMI', '5IGX', '5LEE', '3KLA', '8UOP', '3SYW', '1YZA', '4X68', '6HXL', '6MM9', '7W6Q', '3SUS', '1MUY', '1Q2Y', '2GRP', '1MOH', '3PML', '1ESX', '4E7V', '8I7X', '2R1U', '2VCA', '3D46', '7YFO', '1Z0P', '1V1A', '8KAK', '8SEI', '5NNL', '4XOS', '4USM', '3ID7', '4TPG', '4FLR', '7WQZ', '4LL8', '6WE3', '1G0D', '3NCA', '6KJE', '1W9A', '6KHM', '4GHT', '3GRU', '8OPS', '3TIF', '3OQF', '8OEA', '5V5N', '6H06', '2P7O', '6KOF', '7X1U', '4MV6', '1UV6', '8OPV', '4Q7H', '2C6X', '7UN4', '3NA5', '8JKO', '6SFR', '8DIP', '4BB3', '1HDN', '1N8P', '6YSL', '1ZHX', '7Y6F', '3GA5', '6NHS', '2HH6', '7LQY', '6P5D', '2EZH', '3B2H', '8UW3', '4FJ6', '7F1O', '3ZUA', '4E2V', '4HJH', '5NUE', '8WY5', '4Y64', '1LMO', '1ZOA', '4UCW', '4R8D', '1Y84', '5OZ6', '2EJF', '6F0G', '5LAD', '4LJD', '5COP', '4RUZ', '4KGV', '5MR1', '4HUJ', '8GW1', '4HZV', '1H7J', '3FO4', '3QXM', '7YS5', '5JQ8', '3MNB', '6DWZ', '1CPG', '7PH1', '8F45', '7PU8', '6ALL', '5PRK', '3B1N', '8PA9', '7FM6', '7AIT', '3SPX', '1H9G', '1LL3', '4FWF', '3T33', '2OYL', '7YU8', '3K08', '1MTQ', '5JHY', '1U6C', '1IED', '6BO3', '3TTP', '4PZA', '3T15', '5H3C', '4G4O', '5MXA', '3MFK', '2ZOT', '3EE7', '2XJ1', '7XHD', '2GSY', '6KCK', '4YSA', '4OOP', '2F8C', '3NRQ', '4R36', '2LCW', '7OGG', '2XE1', '5O7E', '4R5S', '7MWA', '6UY3', '6CE0', '6MO7', '8FM0', '3G35', '4B5F', '4DD7', '1JF0', '5KI6', '9EY6', '1H2Y', '1LLN', '3QKA', '6EED', '4M1F', '1XH4', '8U26', '2DNT', '5CCP', '2BZB', '6EOB', '7VFD', '4KKV', '5B4H', '7ABY', '4L9Z', '5SG7', '6CV7', '6AVS', '1SDT', '1ZYT', '2ML8', '7TXJ', '3CXH', '6AY7', '1XVV', '1GBX', '6K3J', '1I6O', '6WA9', '2ZZB', '7UL5', '6JCG', '4I1D', '4AJT', '7TFV', '3AXK', '1W1H', '2R51', '7ZSL', '7UT7', '8PGT', '5OP5', '2K86', '5ZCK', '4F5Z', '7AD0', '2SPT', '4DWU', '1YS3', '4N1G', '7XBB', '2FF4', '3J9B', '5QDW', '2YTS', '5BW5', '1CFL', '2PUK', '1GLO', '5EMW', '3HNC', '6XXM', '7W3Q', '3GGM', '1MHH', '3UKQ', '4HHH', '7LJL', '1JHR', '7C7F', '5X2B', '6BGV', '6QKZ', '6CA0', '1KZA', '7DOP', '2V6M', '1WZ9', '3CIZ', '1R6F', '5QB2', '2Z6Y', '4D72', '4XOO', '2I0H', '1Z4X', '4KN9', '5OQA', '1FU5', '6DH7', '1IEW', '1WC9', '5HV4', '6BA5', '7BR2', '4CJE', '1YP2', '2Y9G', '4OK9', '1NA5', '2CHI', '3KR0', '8PHV', '4F12', '1P7S', '5NBM', '1RJ8', '4BEJ', '1WCM', '5JBR', '7TWJ', '3A4X', '1KHF', '7DQ9', '8CFK', '3T9M', '7ZD4', '8AQ4', '6JBN', '5S8F', '3N97', '5MAV', '4YCA', '4HJV', '1BAJ', '4N56', '3EV5', '6HLO', '3QKY', '9EM4', '6P42', '6VZH', '4HY2', '4A0R', '3VHW', '9F18', '6JFR', '7EMV', '3FCY', '8PXB', '4FRR', '8JI8', '2D8E', '4OG2', '6EF2', '7MAA', '2M4Y', '6GPW', '4USD', '4M7Z', '6LA5', '1MKF', '4CM5', '2QB1', '6TL7', '1GZE', '7A1B', '7ZU3', '3H7M', '4XYB', '5ZUI', '3LT7', '2AEU', '3NWX', '2P5D', '1KS3', '2A39', '7V9N', '3K0X', '1QY9', '3C3O', '8SKR', '3ODF', '4J98', '6J4H', '1ASA', '3PLI', '6VIY', '7VR9', '3TYS', '7XPZ', '8ODS', '7TSW', '7UFO', '4ORD', '2E1M', '7DVK', '1Z42', '2N8L', '3CCN', '4EBW', '4GD0', '1ZOM', '5ZF4', '6NZN', '4U31', '5RF3', '5IIM', '7GEW', '2AYX', '6M1U', '3EIV', '2EP4', '2SXL', '7CV7', '5F79', '5X5P', '3HYK', '8GZV', '6SDW', '8TLF', '8SWY', '3VXG', '4DNH', '5PVQ', '7FYH', '2MSL', '6X5U', '6IFC', '2Z06', '5LK0', '5NYF', '6ESU', '7A1H', '4DXN', '2X5U', '7NT7', '1PGX', '4H88', '6DCC', '4NAL', '1WM4', '8KDK', '5K7H', '5JAC', '3UTN', '3VWX', '3ZEB', '1LLA', '2MEE', '2HPE', '7S5O', '3IU8', '1T57', '2M61', '3PQJ', '4TTY', '2E01', '5CUM', '3QK0', '6FX2', '5SFU', '1MRN', '6TVE', '4AGM', '7R4P', '5IJS', '4N8C', '8DTF', '3FH2', '3GOJ', '4O5E', '6VBR', '7TFK', '2NYR', '6YP2', '5ALA', '1Z36', '2O96', '2BHU', '4NMZ', '4BSC', '1GLN', '6QV3', '1RGT', '8BN1', '1ZU5', '5TL6', '1LP9', '8E37', '1E63', '5J3N', '2OXU', '6E1Q', '6FV9', '5CRK', '6LLY', '4F0J', '3GMQ', '5KF9', '1J9J', '2BE9', '6QWM', '7JUF', '5L7U', '3SBI', '8HTD', '4IVJ', '6YPP', '7TUF', '6HAA', '1S4B', '2KAE', '6S2Q', '5Z2V', '5PTQ', '7Z22', '4FFX', '3DO0', '8TNG', '3D4M', '5TIH', '1G3F', '1SUE', '3LV8', '3A60', '6Y9Z', '6CIN', '2YLL', '6PVC', '3VC2', '7TYI', '8XCQ', '1G9T', '3MLN', '1U9A', '5JDJ', '2WMA', '7CRW', '7CJ3', '6Z00', '5X6N', '1BDS', '5TRV', '3CIH', '2QRQ', '2VSD', '7MA2', '3T8L', '1F1D', '3BL9', '5V43', '7FOV', '7RMA', '3NQK', '7WR7', '8SCH', '5SLU', '6ONZ', '5VUO', '7Z3H', '6H6X', '6M7U', '2HMW', '3D12', '9FGR', '3II2', '1OAL', '2EX4', '5X58', '3T0E', '8JLQ', '5FZJ', '5R01', '8GZ2', '3T6Q', '3WR9', '7H9B', '8I08', '8OFW', '8ENS', '3QT5', '1WU7', '8P0C', '3G4C', '6EUT', '7RI1', '2E76', '6Y8C', '5O9J', '5RCI', '3ZWF', '3T63', '4LMU', '7ZLH', '2A54', '5B12', '5NWI', '7GHJ', '6F0J', '3PB7', '8TX5', '8CY6', '6DOI', '4RZ4', '4BPV', '2I4Q', '3IAM', '3WUA', '8UC3', '3IID', '5O7F', '4JO2', '7MHZ', '7VW5', '7KPN', '2AEH', '6J1Z', '4Z7P', '3KG2', '3LET', '3LKM', '7SIP', '3T8K', '2BZJ', '1Q2Z', '7BG0', '1QHA', '4WLU', '1SE9', '3TN5', '1NV1', '6G5Y', '2E4F', '3PEL', '5VM4', '5O5B', '6IN4', '3Q3D', '5HEY', '7KT2', '7LVA', '5JHW', '3S9M', '6UPB', '1JN7', '7FZE', '4II8', '3K5H', '1HKU', '3ECV', '1V6R', '1SFP', '2QP8', '3GV8', '1CAH', '2BHL', '4AZJ', '5TLC', '5TEU', '6JVR', '1N0C', '8EYL', '4L7P', '5PBA', '6G4I', '8R56', '1DYE', '7U7M', '2ARP', '1Q50', '8IF8', '3BGY', '5CN6', '7DP3', '5J49', '7KE4', '5QTU', '3I0M', '3GY8', '5YM9', '3KOH', '4CDI', '7ZA1', '4OAS', '4J1V', '3ANR', '6F63', '2G8E', '6ZRB', '3BUI', '6XU9', '8PXZ', '4BC3', '4G3R', '4ACY', '6HMT', '3F0Y', '6U3Z', '8HTS', '4WP5', '1P4I', '7C07', '4K2I', '6ZGT', '2L7I', '8JLN', '3KY8', '7QB9', '4NS2', '2WE9', '2ANI', '7G96', '7YP8', '1MO5', '4YZI', '2DYJ', '4B5P', '3WW5', '6RRY', '6C6U', '3USL', '6FXS', '2LH8', '5J9F', '4Y5X', '5ZNM', '3AYU', '6SIQ', '5UJS', '7ERX', '7JYT', '7E99', '2IQ9', '3HKU', '6LYF', '6HF3', '7NNS', '8UN4', '8DJ2', '4KYY', '3EAX', '3KZD', '3PX8', '8PBF', '4FXB', '3LLU', '4R29', '6CRK', '8GOO', '1XP4', '7FTK', '2E50', '1HJG', '1D5F', '1OSX', '7BSB', '6UYC', '2M4M', '2LDT', '4H0J', '8BJ5', '5R35', '6Z2C', '7AXG', '4Y2C', '2H2P', '8JL8', '4KOC', '1ERD', '1ME8', '7H3A', '2XDM', '6NUA', '3OUF', '2VNI', '4N1R', '6MGY', '4DFB', '1CQ2', '6I9U', '3LMC', '3Q01', '3ICK', '1N02', '7E3C', '6OK4', '6TDV', '4E70', '4C0B', '3WOE', '8XEP', '4C4P', '3V35', '8GQE', '4LP4', '1J7W', '6IVY', '6MQM', '1PVG', '8ZM5', '8H1O', '2X8W', '1F8G', '3AYZ', '6OCX', '7QOU', '5OZD', '5RDJ', '6V2H', '7U1C', '2AMC', '5PX1', '4LCQ', '4FW6', '4O4Z', '3IVG', '3RY4', '5KQX', '1OFP', '5AEG', '3W76', '2AOQ', '4PNZ', '6JQ3', '2DO7', '6J4J', '1LJY', '5J0C', '4BL1', '6LSZ', '6CUQ', '3R9F', '7SZT', '7M15', '5RTJ', '1C5S', '8KB6', '1QN8', '1O2N', '6KNW', '8H4P', '5OOK', '6GR5', '5CLT', '1GZG', '6CMZ', '5JVN', '1HEI', '5HLP', '2B9C', '4CBO', '1P88', '5NTB', '7WMU', '3DWK', '2Z2F', '3TKH', '5SQI', '4JLO', '5APZ', '5JXL', '4J08', '2MJB', '6DWD', '7LBC', '8TPC', '5SRT', '4EST', '1XYW', '6W2H', '1E9X', '2DM6', '8HYI', '3OH3', '4Q6Q', '4XNB', '1YOZ', '5HYW', '3UK4', '2MND', '2PIH', '2YZJ', '7QYC', '5MRS', '1LUA', '5KEN', '6B0L', '6CSV', '2HT3', '8I6Z', '4LGX', '3NLY', '8U5Z', '3R6O', '8OTN', '1KO5', '7D5C', '1K5M', '4U0K', '2AOA', '3M6Z', '2GUI', '129L', '7UD7', '5NAE', '7G12', '5E53', '4FBZ', '3O2C', '3CQW', '6DNZ', '2DS8', '2D4W', '6QLS', '3R3M', '5YJ4', '4PN1', '4UEQ', '1NGH', '9EQF', '4KYA', '4NFO', '4BLI', '4KAO', '1HF3', '6EIY', '5HAB', '1Y1F', '4OIT', '5WSN', '5F8G', '5YU3', '1P0B', '3ZY5', '6WPS', '8AYO', '5LYX', '7VDQ', '3RJD', '1R86', '1INX', '1TMM', '3LCO', '2ONH', '8Q2R', '5L3Y', '3UGG', '6EK9', '6DG1', '3VFN', '5ZRD', '1MOY', '6T1M', '5OLD', '7XCQ', '1G2P', '2HR6', '4RS7', '6Z8G', '3NKQ', '8CRQ', '3GXV', '1IP2', '2XXD', '8B19', '3Q2N', '3VJA', '7KK2', '6BEM', '3N63', '3XIM', '6U2D', '1WTV', '6P25', '5YTA', '5KI0', '5JJ1', '3OMS', '7W01', '4OV6', '3WLP', '3KW8', '6CWX', '4H1O', '1O99', '1Q1U', '1XLC', '1M8M', '2LF8', '5JDM', '6M68', '5MRL', '8BRZ', '8QN9', '1BX2', '6WG8', '4QKH', '7GKW', '5L3S', '6I8F', '8OTX', '1HFC', '5D7O', '7C4S', '1DFJ', '4EFV', '5DW0', '1QF4', '7ICO', '2GKP', '2DRJ', '5XR0', '4IJC', '2ASB', '5IPP', '2YG1', '3EL1', '4L0J', '8D54', '8BOK', '6HTO', '4U7U', '1WS0', '4CQV', '3II1', '6XRY', '4ZUK', '7MEZ', '7AVA', '1FKV', '4IJ8', '4PMS', '3QH5', '6A44', '1U00', '6M2H', '8DKO', '3V85', '2NX7', '1PBD', '1WLF', '234L', '8K1S', '143L', '3SDG', '1XCJ', '5H7W', '2ZOD', '4HI0', '5QBC', '3G02', '1YIO', '8G1P', '3NKH', '4JGS', '2F2B', '3TPT', '7W0A', '1P0N', '3EE2', '3USY', '5OCB', '8BZJ', '5S9L', '3FLE', '4QUG', '6F4S', '1T2F', '1NI6', '5LE6', '5SPJ', '6DVK', '7FV4', '5YMW', '1JP3', '8DYR', '3WXY', '2UZA', '3HHB', '7R8S', '4R0K', '7SLC', '6NAN', '2YL9', '5C36', '8DP0', '2CVW', '8DVH', '6E5D', '7KYX', '3BAI', '6TME', '3KMS', '7LTU', '5PVJ', '7GOQ', '1QHQ', '1BMZ', '1UNL', '3GWN', '3CHU', '4XFP', '4EPH', '7A2B', '2YQE', '7NIX', '1FU4', '2MSX', '4GXT', '7LX9', '1VEG', '6FU1', '6LJ2', '2W7X', '7WFR', '1LZH', '7DY5', '2AOC', '7ARC', '3NWY', '4EKK', '5PXH', '4Q24', '6OJA', '1URM', '3RUT', '3EH9', '2HH8', '5E5A', '1I7S', '1K8Q', '3QCX', '1J88', '6DBF', '1Q85', '3D5E', '4N9R', '1EHZ']


# # Test for duplicates
def duplicates(pdb):
    tmp = []
    count = 0
    for id in pdb:
        if id in tmp:
            print(id)
            print(count)
        else:
            count += 1
            tmp.append(id)
    if len(tmp) == len(pdb):
        print("No duplicates found")
    return tmp

def count_list(pdb):
    count = 0
    for id in pdb:
        count += 1
    print("Count of list is: " + str(count))

def remove_invalid(pdb, e_list):
    tmp = []
    for id in pdb:
        if id not in e_list:
            tmp.append(id)
    print(tmp)

# remove_invalid(pdb_ids, ['8KGT'])
# count_list(pdb_ids)
# duplicates(pdb_ids)

# Doesn't exist: 1p6z, 1t5v, 2o1r, 2s5v, 2t6w, 2u7x, 3b2b,...
# Discrepancy list: The following has some error, check them out later to see if the model isn't accurate enough, but overall it works well
# 1p6x, 1t8w, 1q8z, 2b8e, 2c9f, 2l8o, 2p2s, 2w9z, 3a1a, 3k1k, 3l2l, 3m3m, 3o5o, 3u1u, 3w3w, 4b2b, 4e5e, 4q7q, 4y5y, 5h8h, 3r2c, 2b4d, 5c67
# ------------------------------------------------------------------------------------------------------------------------------------

# Create directory for PDB files
pdb_dir = 'pdb_files'
os.makedirs(pdb_dir, exist_ok=True)

# Function to suppress output
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Function to download PDB files
def download_pdb(pdb_id, pdb_dir='pdb_files'):
    pdbl = PDBList()
    with suppress_output():
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')

# ------------------------------------------------------------------------------------------------------------------------------------

# seq1 import from biopython has inbuilt function which serves the same purpose
# # Function to convert three-letter codes to single-letter codes
# def three_to_one(three_letter_code):
#     three_to_one_dict = {
#         'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
#         'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
#         'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
#         'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
#     }
#     return three_to_one_dict.get(three_letter_code, 'X')  # 'X' for unknown residues

def parse_sequences(file_path, flag):
    sequences = {}
    residue_to_index = {}
    chain_start_residues = {}
    chain_end_residues = {}
    modres_mapping = {}
    seqadv_residues = {}
    seqadv_deletions = {}

    amino_acids = {
            'ALA', 'CYS', 'ASP', 'GLU', 'PHE',
            'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
            'MET', 'ASN', 'PRO', 'GLN', 'ARG',
            'SER', 'THR', 'VAL', 'TRP', 'TYR',
            # Non-standard amino acids
            'MSE', 'PCA', '9FB', 'ACE'
    }

    # The decrepency list is used to store the file path of the PDB file that has a discrepancy between the HELIX/SHEET records and the SEQRES records i believe. 
    # And is necessary for build_dataset function to get all datasets with decrepency.
    # discrepancy_list = [] # Will return empty if a file has no discrepancy

    def three_to_one(residue):
        try:
            if residue == 'PCA':
                return 'Q'
            if residue == '9FB':
                return 'P'
            return seq1(residue)
        except:
            return 'X'
        
    def is_nucleotide_sequence(sequence):
        # List of 20 standard amino acids
        return all(residue in amino_acids for residue in sequence)
    
    def is_nucleotide_residue(residue):
        return residue in amino_acids
    
    def has_modres_residue(sequence):
        flag = False
        for res in sequence:
            if res in modres_mapping.keys():
                flag = True
                break
        return flag
    
    def update_dict(seqadv_dict, dict):
        new_dict = {}
        temp = 0
        temp_list = []

        for chain_id, seqadv_list in seqadv_dict.items():
            if new_dict.get(chain_id) == None:
                new_dict[chain_id] = {}
            temp = len(seqadv_list)
            temp_list = seqadv_list
            residue_list = []
            index_list = []
            count = 0

            for residue, index in dict[chain_id].items():
                first_residue = residue
                break

            flag = False
            if temp_list[0] < first_residue:
                flag = True
            
            for residue, index in dict[chain_id].items():
                if temp_list == [] or (residue_list != [] and temp_list[0] > residue_list[-1]):
                    if residue_list != []:
                        for i in index_list:
                            if flag:
                                new_dict[chain_id][residue_list[0]] = index + i
                            else:
                                new_dict[chain_id][residue_list[0]] = i + temp
                            residue_list.pop(0)
                        if flag:
                            new_dict[chain_id][residue] = index + i + 1
                        else:
                            new_dict[chain_id][residue] = index + temp
                        count = i + 1
                    else:
                        if flag:
                            new_dict[chain_id][residue] = index + count
                        else:
                            new_dict[chain_id][residue] = index + temp

                elif temp_list != [] and temp_list[0] > residue:
                    if flag:
                        new_dict[chain_id][residue] = index + count
                    else:
                        new_dict[chain_id][residue] = index

                elif temp_list != [] and temp_list[0] < residue:
                    if flag:
                        new_dict[chain_id][temp_list[0]] = index + count
                    else:
                        new_dict[chain_id][temp_list[0]] = index
                    temp_list.pop(0)
                    residue_list.append(residue)
                    index_list.append(index)

            if temp_list != []:
                for residue in temp_list:
                    index += 1
                    new_dict[chain_id][residue] = index + count
                temp_list = []
                    
        for chain_id, residue_dict in dict.items():
            if chain_id not in new_dict:
                new_dict[chain_id] = residue_dict

        return new_dict
                

    def is_in_range(seqadv_dict, start_res_dict, end_res_dict):
        # flag = False
        for chain_id, seqadv_list in seqadv_dict.items():
            # Here residue is the residue number in the SEQADV records, so it implies position not actual residue although position is supposed to be unique
            for residue in seqadv_list:
                for i in range(len(start_res_dict[chain_id])):
                    if start_res_dict[chain_id][i] <= residue <= end_res_dict[chain_id][i]:
                        return True     
        return False  

    def handle_deletions(seqadv_deletions, residue_to_index):
        new_dict = {}
        for chain_id, deletion_list in seqadv_deletions.items():
            for residue in deletion_list:
                residue_to_index[chain_id].pop(residue, None)
        
        return residue_to_index
    
    def update_dict_indices_after_deletions(residue_to_index):
        new_dict = {}
        count = 0
        for chain_id, residue_dict in residue_to_index.items():
            count = 0
            new_dict[chain_id] = {}
            for residue_index, index in sorted(residue_dict.items()): 
                new_dict[chain_id][residue_index] = count
                count += 1
        return new_dict
       

    # def aggregate_dict(start_residue, ):


    with open(file_path, 'r') as file:
        # print(file_path)
        for line in file:
            try:
                # The DBREF has the start and end residue numbers for each chain, which is for mapping the residue numbers to the sequence indices
                if line.startswith('DBREF'):
                    chain_id = line[12]
                    if flag == False:
                        start_residue = int(line[55:61].strip())
                        end_residue = int(line[62:68].strip())
                    else:
                        start_residue = int(line[14:18].strip())
                        end_residue = int(line[20:24].strip())
                    if chain_id in chain_start_residues:
                        chain_start_residues[chain_id].append(start_residue)
                        chain_end_residues[chain_id].append(end_residue)
                    else:
                        chain_start_residues[chain_id] = [start_residue]
                        chain_end_residues[chain_id] = [end_residue]
                # The SEQADV has extra residues that are not in the SEQRES records
                elif line.startswith('SEQADV'):
                    chain_id = line[16]
                    seqadv_deletion = line[49:70].strip()  # Extract the type of SEQADV (e.g., DELETION, CONFLICT, etc.)

                    # Skip lines with ENGINEERED MUTATION or CONFLICT
                    if seqadv_deletion == 'ENGINEERED MUTATION' or seqadv_deletion == 'CONFLICT':
                        continue

                    if seqadv_deletion == 'DELETION':
                        seqadv_deletion_residue = int(line[44:49].strip()) # For deletion cases, use 44:49 for residue number
                        if chain_id not in seqadv_deletions:
                            seqadv_deletions[chain_id] = []
                        seqadv_deletions[chain_id].append(seqadv_deletion_residue)
                    else:
                        # Handle non-deletion SEQADV (e.g., insertions or modifications)
                        seqadv_residue = int(line[18:22].strip())  # For non-deletion cases, use 18:22 for residue number
                        if chain_id not in seqadv_residues:
                            seqadv_residues[chain_id] = []
                        seqadv_residues[chain_id].append(seqadv_residue)
                
                # # The MODRES has the mapping of modified residues to the original residues
                elif line.startswith('MODRES'):
                    modres = line[12:15].strip()
                    original_residue = line[24:27].strip()
                    modres_mapping[modres] = original_residue
            except ValueError as e:
                # print(f"ValueError occurred while processing line: {line}")
                pass
                
        # print(f"Chain start residues: {chain_start_residues}")
        # print(f"Chain end residues: {chain_end_residues}")
        # print(f"SEQADV residues: {seqadv_residues}")
        # print(f"SEQADV deletions: {seqadv_deletions}")
        # print(f"MODRES mapping: {modres_mapping}")

        file.seek(0)  # Reset file pointer to the beginning
        for line in file:
            if line.startswith('SEQRES'):
                chain_id = line[11]
                if chain_id not in sequences:
                    sequences[chain_id] = []
                    residue_to_index[chain_id] = {}
                parts = line.split()
                residues = parts[4:]

                # Check if there are no residues in the chain and skip processing if empty
                if len(residues) == 0:
                    continue

                if is_nucleotide_sequence(residues):
                    residues = [three_to_one(res) for res in residues]

                elif has_modres_residue(residues):
                    temp = []
                    for res in residues:
                        if is_nucleotide_residue(res):
                            temp += three_to_one(res)
                        elif res in modres_mapping.keys():
                            temp += three_to_one(modres_mapping[res])  
                    residues = temp
                else:
                    # print(f"Nucleotide or non-protein sequence found in chain {chain_id}, skipping...")
                    continue
                
                seq_position = 0
                # Update residue to index mapping
                for i in range(len(chain_start_residues[chain_id])):
                    start_residue = chain_start_residues[chain_id][i]
                    end_residue = chain_end_residues[chain_id][i]

                    for residue_number in range(start_residue, end_residue + 1):
                            residue_to_index[chain_id][residue_number] = seq_position
                            seq_position += 1
                    
                # print(f"SEQRES: chain={chain_id}, residues={residues}")
    
                sequences[chain_id].extend(residues)

    # print(residue_to_index['A'])

    chains_to_remove = []
    for chain_id in sequences:
        if sequences[chain_id] == []:
            chains_to_remove.append(chain_id)
    
    for chain_id in chains_to_remove:
        del sequences[chain_id]

    # temp = sequences
    # for chain_id in temp:
    #     if temp[chain_id] != []:
    #         temp[chain_id] = ''.join(temp[chain_id])
    # print(temp)


    if seqadv_residues != {} and not is_in_range(seqadv_residues, chain_start_residues, chain_end_residues) and sequences != {}:
        residue_to_index = update_dict(seqadv_residues, residue_to_index)

    if seqadv_deletions != {} and sequences != {}:
        residue_to_index = handle_deletions(seqadv_deletions, residue_to_index)
        residue_to_index = update_dict_indices_after_deletions(residue_to_index)

    # print(residue_to_index['A'])
                
    return sequences, residue_to_index, chain_start_residues, chain_end_residues, seqadv_residues

# ---------------

def map_secondary_structure(file_path, sequences, residue_to_index):
    secondary_structures = {}

    # The decrepency list is used to store the file path of the PDB file that has a discrepancy between the HELIX/SHEET records and the SEQRES records i believe. 
    # And is necessary for build_dataset function to get all datasets with decrepency.
    discrepancy_list = [] # Will return empty if a file has no discrepancy

    def initialize_secondary_structure(chain_id, length):
        secondary_structures[chain_id] = ['C'] * length

    for chain_id in sequences:
        if chain_id not in secondary_structures:
            initialize_secondary_structure(chain_id, len(sequences[chain_id]))

    with open(file_path, 'r') as file:
        for line in file:

            chain_id = ''
            if line.startswith('HELIX') or line.startswith('SHEET'):
                chain_id = line[19] if line.startswith('HELIX') else line[21]
                if chain_id not in sequences:
                    continue
                try:
                    start_residue = int(line[21:25].strip()) if line.startswith('HELIX') else int(line[22:26].strip())
                    end_residue = int(line[33:37].strip())
                    sec_type = 'H' if line.startswith('HELIX') else 'E'
                    # print(f"{'HELIX' if line.startswith('HELIX') else 'SHEET'}: chain={chain_id}, start={start_residue}, end={end_residue}")

                    # Initialize secondary structure if not already done
                    if chain_id not in secondary_structures:
                        initialize_secondary_structure(chain_id, len(sequences[chain_id]))

                    # Debug print to show the mapping of residues
                    # print(f"Residue to index mapping for chain {chain_id}: {residue_to_index[chain_id]}")

                    # Update secondary structure
                    try:
                        for residue_num in range(start_residue, end_residue + 1):
                                if residue_num in residue_to_index[chain_id]:
                                    index = residue_to_index[chain_id][residue_num]
                                    secondary_structures[chain_id][index] = sec_type
                                # print(f"Updating index {index} for chain {chain_id} to {sec_type}")
                                else:
                                    # print(f"Residue number {residue_num} not found in mapping for chain {chain_id}")
                                    if file_path not in discrepancy_list:
                                        discrepancy_list.append(file_path) 
                    except IndexError as e:
                            # print(f"IndexError in chain {chain_id}, start: {start_residue}, end: {end_residue}, residue: {residue_num}")
                            # if file_path not in discrepancy_list:
                            #             discrepancy_list.append(file_path) 
                            continue
                except ValueError as e:
                    print(f"Error parsing {'HELIX' if line.startswith('HELIX') else 'SHEET'} line: {line}")

    return secondary_structures, discrepancy_list

def parse_pdb_multiple_chains(file_path):
    # Step 1: Sequence collection and residue-to-index mapping
    sequences, residue_to_index, chain_start_residues, chain_end_residues, seqadv_residues = parse_sequences(file_path, False)
    
    # Step 2: Secondary structure mapping
    secondary_structures, discrepancy_list = map_secondary_structure(file_path, sequences, residue_to_index)

    if discrepancy_list != []:
        try:
            sequences, residue_to_index, chain_start_residues, chain_end_residues, seqadv_residues = parse_sequences(file_path, True)
            secondary_structures, discrepancy_list = map_secondary_structure(file_path, sequences, residue_to_index)
        except:
            sequences, residue_to_index, chain_start_residues, chain_end_residues, seqadv_residues = parse_sequences(file_path, False)
            secondary_structures, discrepancy_list = map_secondary_structure(file_path, sequences, residue_to_index)


    if secondary_structures != {}:
        for chain_id in secondary_structures:
            if secondary_structures[chain_id] != []:
                secondary_structures[chain_id] = ''.join(secondary_structures[chain_id])
    
    if sequences != {}:
        for chain_id in sequences:
            if sequences[chain_id] != []:
                sequences[chain_id] = ''.join(sequences[chain_id])
        
    return sequences, secondary_structures, discrepancy_list

# ------------------------------------------------------------------------------------------------------------------------------------
    
def get_random_pdb_id(pdb_ids):
    if not pdb_ids:
        return None
    return random.choice(pdb_ids)

error_list = ['8I2G', '8DIP', '6VBR', '5R01', '6J1Z', '6CSV', '6EIY', '8DKO']

discrepancy_ids =  ['1P6X', '1T8W', '1Q8Z', '2B8E', '2C9F', '2P2S', '2W9Z', '3A1A', '3K1K', '3L2L', '3M3M', '3O5O', '3U1U', '3W3W', '4B2B', '4E5E', 
                    '4Q7Q', '4Y5Y', '5H8H', '3R2C', '2B4D', '5C67', '5N8E', '5O8F', '5Z8Q', '6E8V', '6V9C', '6Z9G', '7N9U', '7T10', '7X14', '6SVB', 
                    '4IQI', '7KJ0', '4L3P', '6HK4', '1KGW', '3VR2', '5DFP', '1W5Z', '6HVA', '1QAJ', '6OS0', '1Z6A', '7MVS', '7FBQ', '2UZS', '2RK9', 
                    '3LJ0', '4TU7', '6M9S', '4CT6', '3DCA', '6MBE', '6FU4', '5YT2', '4QSS', '8GFZ', '4IQV', '4I19', '8D4T', '3ZS1', '7S0R', '2H4E', 
                    '3OQL', '4ZJG', '6BDU', '8SHK', '2VOV', '1VGG', '1IZL', '1S8I', '6HYP', '3PMV', '5Z9P', '1KSL', '6YB1', '5QAF', '2CCF', '3RGH', 
                    '5C4Y', '6AAH', '3GHD', '2OMK', '4V1A', '7AM3', '2VQI', '5AME', '2YMP', '8QAB', '8C5L', '7AJS', '3CI5', '8HA2', '5L80', '2EF7', 
                    '5GIC', '5US3', '7DBI', '1NTP', '3DSN', '5K75', '2QTY', '2Q4T', '8I2G', '1LX7', '5ZCA', '3SCY', '3HJJ', '2PXW', '2OKK', '4S3J', 
                    '6W8L', '4JAW', '1Y9E', '2OBA', '1P1N', '3ZMF', '2BTC', '2XX8', '5KXI', '8F6P', '5Y7Y', '5APH', '5HMA', '3T26', '7AUX', '1IRU', 
                    '7TQU', '1CLY', '1HJK', '7G0I', '3M5M', '3S46', '3MAJ', '1J16', '3OU0', '5PO3', '2WG6', '3BWB', '3W5C', '1S5S', '6HK8', '7R5F', 
                    '4OEX', '2HTD', '7JQR', '2O7H', '8J2J', '1KT2', '7COH', '1TY4', '6OPJ', '1YLE', '8H1W', '5VO5', '4HWV', '5C31', '5JNQ', '6UIE', 
                    '6Z2F', '3EMR', '3CDH', '2IEE', '8K8E', '7SNX', '7QSP', '1UBP', '5F5S', '8G0Q', '5IGX', '4X68', '3PML', '4FLR', '4LL8', '6KJE', 
                    '1W9A', '5V5N', '6H06', '6SFR', '8DIP', '1ZHX', '6NHS', '2HH6', '6P5D', '4FJ6', '7F1O', '4HJH', '5NUE', '4LJD', '4HUJ', '3QXM', 
                    '6DWZ', '7PH1', '5PRK', '7FM6', '3K08', '1MTQ', '5H3C', '2XJ1', '7OGG', '6CE0', '8FM0', '3G35', '1LLN', '8U26', '2BZB', '3CXH', 
                    '1I6O', '6WA9', '7UL5', '4I1D', '3AXK', '2R51', '7AD0', '2SPT', '3J9B', '5EMW', '6XXM', '1MHH', '6BGV', '1WZ9', '5QB2', '2Z6Y', 
                    '4KN9', '5OQA', '7BR2', '8PHV', '5NBM', '4BEJ', '1WCM', '5JBR', '6HLO', '3VHW', '9F18', '4FRR', '4OG2', '7ZU3', '3NWX', '2A39', 
                    '1QY9', '3PLI', '6VIY', '3TYS', '1Z42', '6M1U', '3HYK', '8SWY', '4DNH', '5PVQ', '5NYF', '6ESU', '4DXN', '2X5U', '8KDK', '1T57', 
                    '2M61', '2E01', '5IJS', '3FH2', '6VBR', '6YP2', '2BHU', '8BN1', '1ZU5', '1LP9', '8E37', '5J3N', '4F0J', '4IVJ', '5PTQ', '1SUE', 
                    '5JDJ', '2WMA', '3CIH', '2VSD', '3T8L', '7FOV', '3NQK', '2EX4', '8JLQ', '5R01', '1WU7', '7RI1', '5O9J', '2A54', '5O7F', '6J1Z', 
                    '3T8K', '7BG0', '2E4F', '7KT2', '1N0C', '3I0M', '3GY8', '6F63', '4BC3', '4ACY', '8HTS', '4WP5', '1P4I', '8JLN', '3KY8', '2WE9', 
                    '7G96', '1MO5', '4YZI', '5ZNM', '3AYU', '5UJS', '6LYF', '8GOO', '1XP4', '3OUF', '4DFB', '3LMC', '3Q01', '1N02', '3WOE', '1PVG', 
                    '1F8G', '6V2H', '5J0C', '1C5S', '1O2N', '5OOK', '5APZ', '1E9X', '2PIH', '2YZJ', '6CSV', '4LGX', '1K5M', '3O2C', '3CQW', '6DNZ', 
                    '4KYA', '6EIY', '3LCO', '5L3Y', '4RS7', '3OMS', '6CWX', '1DFJ', '2GKP', '5IPP', '2YG1', '4L0J', '4U7U', '4IJ8', '6A44', '8DKO', 
                    '3NKH', '3FLE', '1JP3', '4R0K', '2YL9', '3BAI', '5PVJ', '7NIX', '4GXT', '4EKK', '3QCX', '1Q85']

discrepancy_ids2 = ['1T8W', '1Q8Z', '2B8E', '2C9F', '2P2S', '2W9Z', '3K1K', '3L2L', '3M3M', '3U1U', '4B2B', '4E5E', '4Q7Q', '4Y5Y', '3R2C', '2B4D', 
                    '5C67', '5N8E', '5O8F', '6E8V', '6G8X', '6O95', '6Z9G', '7N9U', '7X14', '8H1E', '6SVB', '4IQI', '8BX6', '7KJ0', '4L3P', '6HK4', 
                    '1KGW', '3VR2', '5DFP', '1W5Z', '6HVA', '1QAJ', '6OS0', '1Z6A', '7FBQ', '2UZS', '2RK9', '6M9S', '3DCA', '6MBE', '4JDI', '6FU4', 
                    '5YT2', '8GFZ', '4I19', '8D4T', '6EIR', '3ZS1', '7S0R', '2H4E', '3OQL', '4ZJG', '6BDU', '2VOV', '1VGG', '1IZL', '1S8I', '6HYP', 
                    '3PMV', '1KSL', '6YB1', '5QAF', '2CCF', '3RGH', '5C4Y', '6AAH', '3GHD', '2OMK', '4V1A', '2VQI', '2YMP', '8QAB', '7AJS', '3CI5', 
                    '5L80', '2EF7', '5GIC', '5US3', '6DAJ', '7DBI', '1NTP', '5K75', '2Q4T', '8I2G', '8AI6', '1LX7', '5ZCA', '3SCY', '3HJJ', '2PXW', 
                    '2OKK', '6W8L', '1Y9E', '7VB6', '2OBA', '3ZMF', '2BTC', '5KXI', '8F6P', '5HMA', '3T26', '7AUX', '1IRU', '7TQU', '1CLY', '1HJK', 
                    '4XOW', '3M5M', '3S46', '3MAJ', '1J16', '3OU0', '2R2O', '4XBU', '3BWB', '3W5C', '1S5S', '2HTD', '7JQR', '8J2J', '1KT2', '7COH', 
                    '1TY4', '6OPJ', '1YLE', '8H1W', '5C31', '5J85', '6UIE', '6Z2F', '3EMR', '3CDH', '2IEE', '8K8E', '7SNX', '1UBP', '8G0Q', '5IGX', 
                    '4LL8', '1W9A', '5V5N', '6H06', '3NA5', '8DIP', '6NHS', '2HH6', '6P5D', '4FJ6', '4HJH', '5NUE', '4UCW', '4LJD', '4HUJ', '3QXM', 
                    '7PH1', '1MTQ', '5H3C', '2XJ1', '7OGG', '6CE0', '8FM0', '3G35', '1LLN', '2BZB', '7ABY', '3CXH', '1I6O', '4I1D', '3AXK', '2R51', 
                    '2K86', '7AD0', '2SPT', '3J9B', '5EMW', '6XXM', '1MHH', '6BGV', '1WZ9', '5QB2', '2Z6Y', '4KN9', '5OQA', '8PHV', '5NBM', '4BEJ', 
                    '1WCM', '5JBR', '6HLO', '3VHW', '9F18', '4OG2', '7ZU3', '3NWX', '2A39', '1QY9', '3PLI', '3TYS', '1Z42', '6M1U', '3HYK', '4DNH', 
                    '5NYF', '4DXN', '2X5U', '8KDK', '7S5O', '1T57', '2M61', '5IJS', '6VBR', '6YP2', '2BHU', '8BN1', '1LP9', '8E37', '5J3N', '4F0J', 
                    '4IVJ', '2WMA', '3CIH', '3T8L', '3NQK', '8JLQ', '5R01', '1WU7', '7RI1', '2A54', '6J1Z', '3T8K', '3TN5', '2E4F', '5TEU', '1N0C', 
                    '3I0M', '3GY8', '3ANR', '4BC3', '4ACY', '8HTS', '1P4I', '8JLN', '3KY8', '2WE9', '1MO5', '4YZI', '6FXS', '3AYU', '6LYF', '8GOO', 
                    '1XP4', '4DFB', '3LMC', '3Q01', '3WOE', '1PVG', '1F8G', '6V2H', '1C5S', '1O2N', '5OOK', '5APZ', '1E9X', '2PIH', '2YZJ', '6CSV', 
                    '4LGX', '1K5M', '3O2C', '3CQW', '6DNZ', '4UEQ', '9EQF', '4KYA', '6EIY', '3LCO', '4RS7', '3OMS', '6CWX', '1DFJ', '2GKP', '2YG1', 
                    '4L0J', '4IJ8', '8DKO', '3NKH', '3FLE', '1JP3', '4R0K', '3BAI', '4GXT', '4EKK', '3QCX', '1Q85']

discrepancy_ids3 = ['1Q8Z', '2C9F', '2W9Z', '3K1K', '3L2L', '4B2B', '4E5E', '4Y5Y', '3R2C', '2B4D', '5C67', '5N8E', '5O8F', '6E8V', '6G8X', '6O95', 
                    '6Z9G', '7N9U', '7X14', '8H1E', '6SVB', '8BX6', '7KJ0', '4L3P', '6HK4', '1KGW', '5DFP', '1W5Z', '6HVA', '1QAJ', '6OS0', '1Z6A', 
                    '7FBQ', '2UZS', '6MBE', '4JDI', '5YT2', '8GFZ', '8D4T', '6EIR', '3ZS1', '2H4E', '2VOV', '1IZL', '1S8I', '6HYP', '3PMV', '6YB1', 
                    '5QAF', '2CCF', '3RGH', '6AAH', '2OMK', '4V1A', '2VQI', '2YMP', '8QAB', '7AJS', '3CI5', '5L80', '5GIC', '5US3', '6DAJ', '1NTP', 
                    '5K75', '8I2G', '8AI6', '5ZCA', '3SCY', '2PXW', '2OKK', '6W8L', '7VB6', '3ZMF', '2BTC', '5KXI', '8F6P', '5HMA', '3T26', '7AUX', 
                    '1IRU', '7TQU', '1CLY', '1HJK', '4XOW', '3M5M', '3S46', '1J16', '3OU0', '4XBU', '3W5C', '1S5S', '7JQR', '8J2J', '1KT2', '7COH', 
                    '6OPJ', '8H1W', '5J85', '6Z2F', '8K8E', '7SNX', '1UBP', '8G0Q', '5IGX', '4LL8', '5V5N', '6H06', '3NA5', '8DIP', '6P5D', '4FJ6', 
                    '4HJH', '5NUE', '4UCW', '4LJD', '3QXM', '7PH1', '1MTQ', '2XJ1', '7OGG', '6CE0', '8FM0', '3G35', '1LLN', '2BZB', '7ABY', '3CXH', 
                    '3AXK', '2K86', '7AD0', '2SPT', '3J9B', '5EMW', '6XXM', '1MHH', '6BGV', '1WZ9', '5QB2', '2Z6Y', '4KN9', '5OQA', '5NBM', '4BEJ', 
                    '1WCM', '6HLO', '3VHW', '9F18', '3NWX', '2A39', '3PLI', '6M1U', '4DXN', '2X5U', '8KDK', '7S5O', '2M61', '5IJS', '6VBR', '6YP2', 
                    '2BHU', '8BN1', '1LP9', '5J3N', '4F0J', '4IVJ', '2WMA', '8JLQ', '5R01', '7RI1', '2A54', '6J1Z', '3TN5', '2E4F', '5TEU', '1N0C', 
                    '3GY8', '3ANR', '8HTS', '1P4I', '8JLN', '1MO5', '4YZI', '6FXS', '3AYU', '8GOO', '3Q01', '6V2H', '1C5S', '1O2N', '5OOK', '5APZ', 
                    '1E9X', '6CSV', '4LGX', '1K5M', '3O2C', '3CQW', '4UEQ', '9EQF', '4KYA', '6EIY', '3LCO', '6CWX', '1DFJ', '2YG1', '4IJ8', '8DKO', 
                    '4R0K', '3BAI', '4EKK', '3QCX']

discrepancy_ids4 = ['1X3A', '3M3M', '3O5O', '5O8F', '6O95', '8F1C', '8Q1N', '5LOM', '1V61', '1QAJ', '3PXN', '3TGV', '7MK3', '1S8I', '6HTI', '2OMK', 
                    '5AME', '7KRB', '8C5L', '7AJS', '7UEQ', '8HA2', '6E18', '3CQ0', '2DM1', '4S3J', '5KXI', '1IRU', '5PO3', '8FDS', '2O7H', '1KT2', 
                    '4HWV', '5JNQ', '2LML', '7QSP', '1KM0', '8G0Q', '4FLR', '4GHT', '8DIP', '3B2H', '5PRK', '3K08', '6KCK', '8U26', '4F5Z', '2YTS', 
                    '5JBR', '4YCA', '4DNH', '5PVQ', '6ESU', '8KDK', '2E01', '3FH2', '1ZU5', '5J3N', '4IVJ', '5PTQ', '3MLN', '2EX4', '8P0C', '5O9J', 
                    '2A54', '5O7F', '7BG0', '5HEY', '4WP5', '6ZGT', '4Y5X', '2M4M', '3OUF', '3Q01', '5J0C', '2B9C', '5L3Y', '7W01', '5IPP', '8BZJ', 
                    '2YL9', '3KMS', '5PVJ', '6FU1']

discrepancy_ids5 = ['1X3A', '3D4D', '3M3M', '3O5O', '3W3W', '4Y5Y', '5O8F', '6M93', '6O95', '8F1C', '5LOM', '1V61', '8DFH', '1QAJ', '3PXN', '5T8U', 
                    '3TGV', '7MK3', '1S8I', '8WLS', '5Z9P', '2OMK', '5UIP', '7KRB', '7AJS', '7UEQ', '6E18', '2DM1', '4S3J', '7B1A', '5KXI', '8AES', 
                    '5APH', '1IRU', '7PEJ', '4JM8', '7XIX', '1KT2', '7P5Y', '7LM7', '4HWV', '5JNQ', '7QSP', '1KM0', '8G0Q', '4FLR', '4GHT', '3B2H', 
                    '7F1O', '3K08', '8U26', '2YTS', '5X2B', '4DNH', '3FH2', '4NMZ', '1ZU5', '5J3N', '4IVJ', '7TYI', '2EX4', '2A54', '3KG2', '1P4I', 
                    '4Y5X', '3OUF', '3Q01', '1MOY', '7W01', '5IPP']

discrepancy_ids6 = ['3D4D', '3O5O', '3W3W', '5H8H', '5C67', '5O8F', '6M93', '1QAJ', '7MVS', '5T8U', '1S8I', '8WLS', '5Z9P', '2OMK', '7AM3', '8C5L', 
                    '1P1N', '2XX8', '1IRU', '7PEJ', '4JM8', '7XIX', '1KT2', '5VO5', '7QSP', '1KM0', '8G0Q', '4FLR', '7F1O', '3K08', '7OGG', '8U26', 
                    '8SWY', '4NMZ', '1ZU5', '5J3N', '4IVJ', '2EX4', '5O9J', '2A54', '3KG2', '7BG0', '1P4I', '3Q01', '5J0C', '1MOY', '5IPP', '6A44']

# count_list(discrepancy_ids)
# count_list(discrepancy_ids2)
# count_list(discrepancy_ids3)
# count_list(discrepancy_ids4)
# count_list(discrepancy_ids5)
# count_list(discrepancy_ids6)
# duplicates(discrepancy_ids6)

# print(get_random_pdb_id(discrepancy_ids6))


# for the files which doesn't have the position in the sequence do a pre-check and add that position. Also look into SEQADV and what it is.
# IF it is meaningful and relevant, then take it into account

# # Test and study discrepency files here 
# result = parse_pdb_multiple_chains("pdb_files/pdb3gxv.ent") #1q8z
# if result is not None:
#     sequences, secondary_structures, discrepancy_list = result
#     print(sequences)
#     print(secondary_structures)
#     print(discrepancy_list)
# else:
#     print("The function returned None.")

# Also test valid sequences to see if the function works well after changes: 1a2t, 1a8m, 1bna, 1c0a, 1d3z, 1e7p, 1f9t, 1gid, 1h9n, 1jff, 1kx5

# discrepancy resolved files: 1p6x
# ------------------------------------------------------------------------------------------------------------------------------------

# 1a8m, 1bna, 1c0a, 1d3z, 1e7p, 1f9t, 1gid, 1h9n, 1jff, 1kx5
# # Example usage
# file_path = 'pdb_files/pdb1f9t.ent'
# sequences, secondary_structures = parse_pdb_multiple_chains(file_path)
# for chain_id in sequences:
#     print(f'Chain {chain_id}:')
#     print('Sequence:', sequences[chain_id])
#     print('Secondary Structure:', secondary_structures[chain_id])
# print(sequences)

# # Verify if the secondary structure annotations match the expected HELIX and SHEET records
# for chain_id in sequences:
#     print(f'Chain {chain_id} Length: {len(sequences[chain_id])}')
#     print(f'Secondary Structure Length: {len(secondary_structures[chain_id])}')
#     assert len(sequences[chain_id]) == len(secondary_structures[chain_id]), "Length mismatch between sequence and secondary structure"
#     print(f'First 100 residues of Chain {chain_id}: {sequences[chain_id][:100]}')
#     print(f'First 100 annotations of Secondary Structure of Chain {chain_id}: {secondary_structures[chain_id][:100]}')

# ------------------------------------------------------------------------------------------------------------------------------------

# Function to preprocess sequences (primary)
def preprocess_sequences(sequence, max_length=700):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    encoding = np.zeros((max_length, len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_length]):
        if aa in aa_dict:
            encoding[i, aa_dict[aa]] = 1
    return encoding

## Testing the function
# print(preprocess_sequences('ACDEFGHIKLMNPQRSTVWYAAAA', 24))

# Function to decode one-hot encoded sequences
def decode_sequence(one_hot_matrix):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''
    for row in one_hot_matrix:
        index = np.argmax(row)
        if np.sum(row) > 0:  # Ensure there's a non-zero row
            sequence += amino_acids[index]
        else:
            break  # Stop decoding when you encounter the first all-zero row
    return sequence

# Decode the first sequence
# decoded_sequence_1A8M = decode_sequence(X[0])
# print(f"Decoded sequence for '1A8M': {decoded_sequence_1A8M}")
# print(f"First 10 residues of '1A8M': {decoded_sequence_1A8M[:10]}")
# print(f"Length of the sequence for '1A8M': {len(decoded_sequence_1A8M)}")

# one_hot_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

# # Decode the sequence
# decoded_sequence = decode_sequence(one_hot_matrix)
# print(decoded_sequence)  
# #VRSSS

# # Example: Print the first sequence to inspect
# print("First sequence (X[0]):")
# print(X[0])

# Example: Print a specific part of the first sequence to inspect
# print("First 5 residues of the first sequence (X[0][:5]):")
# print(X[0][:5])

# ------------------------------------------------------------------------------------------------------------------------------------

# Function to preprocess secondary structures (secondary sequences)
def preprocess_structures(structure, max_length=700):
    structure_dict = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
    encoding = np.zeros((max_length, 3), dtype=np.float32)
    for i, s in enumerate(structure[:max_length]):
        if s in structure_dict:
            encoding[i, structure_dict[s]] = 1
    return encoding


## Tests for secondary structure encoding and decoding

# struct = 'HHHCCEECHEHCHCCCC'
# print(preprocess_structures(struct, 17))

# Function to decode the encoded secondary structures back to the string representation
def decode_structures(encoding):
    index_to_structure = {0: 'H', 1: 'E', 2: 'C'}
    structure = []
    for row in encoding:
        index = np.argmax(row)
        structure.append(index_to_structure.get(index, 'C'))
    return ''.join(structure)

# # Test the encoding and decoding process
# def test_preprocess_and_decode():
#     # Define some test cases
#     test_cases = [
#         "HHHEEECCC",
#         "CCHHHEECCCCHHHEE",
#         "ECCCCCHHHHEEE",
#         "HHCCHHEECC",
#         "HHHHHHHHHH",
#         "EEEEEEEEEE",
#         "CCCCCCCCCC",
#         "HHHEECCCHHEECC"
#     ]
    
#     # Maximum length for encoding
#     max_length = 18

#     # Run the tests
#     for case in test_cases:
#         print(f"Original structure: {case}")
#         encoded = preprocess_structures(case, len(case))
#         print("Encoded structure: ")
#         print(f"{encoded}")
#         decoded = decode_structures(encoded[:len(case)])  # Only decode the part of the array that was used
#         print(f"Decoded structure:  {decoded}")
#         assert case == decoded, f"Test failed for case: {case}"
#         print("Test passed!\n")

# # Run the test function
# test_preprocess_and_decode()

# ------------------------------------------------------------------------------------------------------------------------------------
# change max length to 300, 350. But for now 400 is fine.
# Consider chaching the files to avoid downloading them again and again.
# Function to build the dataset
def build_dataset(pdb_ids, pdb_dir='pdb_files', max_length=400):
    X, y = [], []
    error_list = []
    discrepancy_list = []
    for pdb_id in pdb_ids:
        try:
            download_pdb(pdb_id, pdb_dir)
            file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
            result = parse_pdb_multiple_chains(file_path)
            if result is not None:
                sequences, secondary_structures, d_list = result
            elif result is None:
                continue
            if d_list != []:
                discrepancy_list.append(d_list[0])
            for chain_id in sequences:
                X.append(preprocess_sequences(sequences[chain_id], max_length))
                y.append(preprocess_structures(secondary_structures[chain_id], max_length))
        except Exception as e:
            # print(f'Error processing {pdb_id}: {e}')
            error_list.append(pdb_id)
    return np.array(X), np.array(y), error_list, discrepancy_list

# # Build the dataset
# X, y, error_list, discrepancy_list = build_dataset(pdb_ids)
# print(f'Dataset shape: X={X.shape}, y={y.shape}')
# print("\n" + "Error List:")
# print(error_list)
# print(count_list(error_list))

# sequence_length = X.shape[1]
# num_features = X.shape[2]
# num_classes = y.shape[2]

# 13 to 17 or -8 to -4
def convert_decrepency_list(discrepancy_list):
    tmp = []
    for file in discrepancy_list:
        tmp.append(file[-8:-4].upper())
    return tmp

# print("\n" + "Discrepency List:")
# discrepancy_list = convert_decrepency_list(discrepancy_list)
# print(discrepancy_list)
# print(count_list(discrepancy_list))

# ------------------------------------------------------------------------------------------------------------------------------------

# function to analyze sequence lengths with graph representation using matplotlib
def analyze_sequence_lengths(pdb_ids, pdb_dir='pdb_files'):
    lengths = []
    for pdb_id in pdb_ids:
        try:
            file_path = os.path.join(pdb_dir, f'pdb{pdb_id.lower()}.ent')
            result = parse_pdb_multiple_chains(file_path)
            # If parse_pdb_multiple_chains returns more than two values, 
            # you can unpack them like this
            sequences = result[0]

            for chain_id in sequences:
                lengths.append(len(sequences[chain_id]))
        except Exception as e:
            print(f'Error processing {pdb_id}: {e}')
    return lengths

# # Analyze lengths
# sequence_lengths = analyze_sequence_lengths(pdb_ids)

# # Plot the distribution of sequence lengths
# plt.hist(sequence_lengths, bins=20)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.title('Distribution of Protein Sequence Lengths')
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------
# Try this out, and figure out how to make it work after you have a well working model

# class CosineAnnealingWarmRestarts(Callback):
#     def __init__(self, initial_lr, T_max, eta_min=0, T_mult=2):
#         super(CosineAnnealingWarmRestarts, self).__init__()
#         self.initial_lr = initial_lr
#         self.T_max = T_max
#         self.eta_min = eta_min
#         self.T_mult = T_mult
#         self.current_epoch = 0
#         self.current_cycle = 0

#     def on_epoch_end(self, epoch, logs=None):
#         self.current_epoch += 1
#         T_i = self.T_max * (self.T_mult ** self.current_cycle)
#         if self.current_epoch >= T_i:
#             self.current_cycle += 1
#             self.current_epoch = 0

#         lr = float(self.eta_min + (self.initial_lr - self.eta_min) * (1 + math.cos(math.pi * self.current_epoch / T_i)) / 2)
#          # Ensure learning rate is properly set depending on Keras version
#         if hasattr(self.model.optimizer, 'learning_rate'):
#             tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
#         elif hasattr(self.model.optimizer, 'lr'):
#             tf.keras.backend.set_value(self.model.optimizer.lr, lr)
#         else:
#             raise AttributeError('Optimizer does not have learning rate attribute')


# # Applying the callback
# T_max = 10  # Number of epochs before each restart
# initial_lr = 0.001    # Starting learning rate

# cosine_annealing = CosineAnnealingWarmRestarts(initial_lr=initial_lr, T_max=T_max)

# ------------------------------------------------------------------------------------------------------------------------------------


# # CNN-LSTM hybrid model

# # Updated Model Architecture
# model = Sequential()

# # Input layer
# model.add(Input(shape=(sequence_length, num_features)))

# # CNN layers
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.002)))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.2))  # Add Dropout

# model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.002)))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.2))  # Add Dropout

# model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.002)))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.2))  # Add Dropout

# model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.002)))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Dropout(0.2))  # Add Dropout
# model.add(BatchNormalization())

# # LSTM layers
# model.add(LSTM(128, return_sequences=True, dropout=0.4)) 
# model.add(LSTM(128, return_sequences=True, dropout=0.4))
# model.add(LSTM(128, return_sequences=True, dropout=0.4))

# # Dense layers for classification
# model.add(TimeDistributed(Dense(64, activation='relu', kernel_regularizer=l2(0.002))))  # TimeDistributed to apply Dense layer to each time step
# model.add(Dropout(0.6))
# model.add(TimeDistributed(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.002))))  # Use softmax for multi-class classification and TimeDistributed for final output

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.00015), loss='categorical_crossentropy', metrics=['accuracy'])

# # Summary of the model
# model.summary()


# # ------------------------------------------------------------------------------------------------------------------------------------

# # Split the data into training and validation sets
# split_index = int(0.8 * len(X))
# X_train, X_val = X[:split_index], X[split_index:]
# y_train, y_val = y[:split_index], y[split_index:]

# # Add model checkpoint to save the best model in Keras format
# checkpoint = ModelCheckpoint('best_model_v2.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# # Add early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Implement learning rate scheduling
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping, lr_scheduler])

# # Evaluate the model on the validation set
# loss, accuracy = model.evaluate(X_val, y_val)
# print(f'Validation Loss: {loss}')
# print(f'Validation Accuracy: {accuracy}')

# # Create a figure with 3 subplots
# plt.figure(figsize=(18, 6))

# # Subplot 1: Accuracy
# plt.subplot(1, 3, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# # Subplot 2: Loss
# plt.subplot(1, 3, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')

# # Subplot 3: Error Rate (1 - Accuracy)
# plt.subplot(1, 3, 3)
# plt.plot(1 - np.array(history.history['accuracy']), label='Training Error')
# plt.plot(1 - np.array(history.history['val_accuracy']), label='Validation Error')
# plt.xlabel('Epoch')
# plt.ylabel('Error Rate')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Error Rate')

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Display the combined plot
# plt.show()

# # Load the saved model from the file
# loaded_model = load_model('best_model_v2.keras') # latest model here

# # loaded_model = load_model('best_model.keras') # the 73% accuracy model i believe is saved here

# ------------------------------------------------------------------------------------------------------------------------------------

# best_model.keras got accuracy of 0.73.
# best_model_v2.keras got accuracy of 0.72-0.73

# # Example prediction using the loaded model
# # Make predictions
# predictions = loaded_model.predict(X_val)

# # Convert predictions to class labels
# predicted_labels = np.argmax(predictions, axis=-1)

# # Convert actual labels to class labels (if they are one-hot encoded)
# actual_labels = np.argmax(y_val, axis=-1)

# # Example: Print the first 10 predictions and corresponding actual labels
# for i in range(10):
#     print(f"Predicted: {predicted_labels[i]}")
#     print(f"Actual: {actual_labels[i]}")
#     print("---")  # Adding a separator for clarity

# # Calculate accuracy
# accuracy = accuracy_score(actual_labels.flatten(), predicted_labels.flatten())
# print(f"Accuracy: {accuracy:.2f}")
