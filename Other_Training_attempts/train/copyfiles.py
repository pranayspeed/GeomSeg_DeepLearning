

import os

def get_all_test_cases(test_cases_path):
    test_cases = []

    for seq_index in range(len(os.listdir(test_cases_path))):
        cases_list = os.listdir(test_cases_path+"/"+ f"{seq_index:03d}")
        subtest_count = sum('.npz' in s for s in cases_list)
        for i in range(subtest_count):
            try:
                #test_cases.append(get_test_case_data(test_cases_path, seq_index, i))
                test_cases.append([test_cases_path, seq_index, i])
            except:
                break
    return test_cases


def main():

    source_path = "/home/pranayspeed/Downloads/TRAIN-20s/" # "../data/TRAIN-20s/"
    target_path ="/home/pranayspeed/Downloads/TRAIN-20s-normals/" # "../data/TRAIN-20s-normals/"
    test_cases_paths = get_all_test_cases(source_path)
    for test_case_path in test_cases_paths:
        source_file_path = source_path + f"{test_case_path[1]:03d}" + "/" +f"{test_case_path[2]:05d}" +".prim"
        target_file_path = target_path + f"{test_case_path[1]:03d}" + "/" +f"{test_case_path[2]:05d}" +".prim"
        if not os.path.exists(target_path + f"{test_case_path[1]:03d}" + "/"):
            os.makedirs(target_path + f"{test_case_path[1]:03d}" + "/")
        os.system(f"cp {source_file_path} {target_file_path}")
        print(f"cp {source_file_path} {target_file_path}", end='\r')
        

if __name__ == "__main__":
    main()