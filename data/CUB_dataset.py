

def attribute_read(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        concept_list = []
        for line in content.split('\n'):
            if line:
                i=line.split(' ')[-1]
                concept = i.split('::')[0]

                if concept not in concept_list:
                    concept_list.append(concept)
        
        print(len(concept_list), concept_list)
        return content
    except FileNotFoundError:
        return f"文件 {file_path} 未找到。"
    except Exception as e:
        return f"读取文件 {file_path} 时发生错误：{e}"


def split_dataset(dataset_path):
    

if __name__ == '__main__':
    attribute_read("X:\\pervasive_group\\Shared\\CUB_200_2011\\attributes.txt")
    split_dataset()