import re

class Module_Query_Parser():

    def __init__(self, name='Method_Query_Parser', description='query parser'):
        self.name = name
        self.description = description

    def parse_function_calls(self, input_string):
        function_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^\]]*)\)'
        parameter_set_pattern = r'{([^{}]*)}'
        function_calls = re.findall(function_call_pattern, input_string)

        parsed_function_calls = []
        for function_call in function_calls:
            function_name, parameters_str = function_call
            set_parameter_placeholder_dict = {}
            set_parameters = re.findall(parameter_set_pattern, parameters_str)
            for set_parameter in set_parameters:
                set_parameter_placeholder_key = "set_parameter_{}_placeholder".format(len(set_parameter_placeholder_dict))
                set_parameter_value = '{'+set_parameter+'}'
                set_parameter_placeholder_dict[set_parameter_placeholder_key] = set_parameter_value
                parameters_str = parameters_str.replace(set_parameter_value, set_parameter_placeholder_key)
            parameter_list = []
            nested_count = 0
            nested_function_call_str = ''
            for parameter in parameters_str.split(','):
                parameter = parameter.strip().replace('"', '')
                if '(' in parameter or ')' in parameter:
                    count_left_parentheses = parameter.count('(')
                    count_right_parentheses = parameter.count(')')
                    nested_count = nested_count + count_left_parentheses - count_right_parentheses
                    if nested_function_call_str == '':
                        nested_function_call_str += parameter
                    else:
                        nested_function_call_str += ','+parameter
                    if nested_count == 0:
                        parameter_list.append(self.parse_function_calls(nested_function_call_str))
                        nested_function_call_str = ''
                else:
                    if nested_count == 0:
                        parameter_list.append(parameter)
                    else:
                        nested_function_call_str += ','+parameter
            parameter_list = self.update_parameter_list(parameter_list, set_parameter_placeholder_dict)
            parsed_function_calls.append((function_name, parameter_list))

        if len(parsed_function_calls) == 1:
            return parsed_function_calls[0]
        else:
            return parsed_function_calls

    def update_parameter_list(self, parameter_list, set_parameter_placeholder_dict):
        updated_parameter_list = []
        for parameter in parameter_list:
            if type(parameter) is str:
                if parameter in set_parameter_placeholder_dict:
                    updated_parameter_list.append(set_parameter_placeholder_dict[parameter])
                else:
                    updated_parameter_list.append(parameter)
            elif type(parameter) in [tuple, list]:
                updated_parameter_list.append(self.update_parameter_list(parameter, set_parameter_placeholder_dict))
            else:
                updated_parameter_list.append(parameter)
        if type(parameter_list) is tuple:
            return tuple(updated_parameter_list)
        else:
            return updated_parameter_list

    def parse_output_variable(self, input_string):
        output_variable_pattern = r'\)\s*([-->]*)([a-zA-Z0-9_\s]*)\s*\]'
        output_variables = re.findall(output_variable_pattern, input_string)
        output_variable_list = []
        for output_variable in output_variables:
            if '-->' in output_variable:
                output_variable_list.append(True)
            else:
                output_variable_list.append(False)
        return output_variable_list

    def parse_query(self, query_input):
        function_parameters = self.parse_function_calls(query_input)
        output_variable = self.parse_output_variable(query_input)
        return function_parameters, output_variable

if __name__ == "__main__":
    #input = 'Output: Nodes [ GR ( GL ( "dodecahedral_graph", GB("a", GS("c"))), "periphery" )  -->  r ] have the largest eccentricity [GR(GL("dodecahedral_graph"), "eccentricity")] in the dodecahedral graph, which make them part of its periphery.'
    #input = 'Output: Nodes [GR(GL("gpr", {"dodecahedral_graph", "bull_graph"}, { ( "a", "b" ), (alice, bob) }), "periphery")-->r] have the largest eccentricity [GR(GL("gpr", "dodecahedral_graph"), "eccentricity")] in the dodecahedral graph, which make them part of its periphery. Reasoning Result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] <||> {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5}.'
    input = 'Output: Nodes [GR(GL("cora"), "graph-bert:topic", {paper#1})-->r] have the largest eccentricity [GR(GL("gpr", "dodecahedral_graph"), "eccentricity")] in the dodecahedral graph, which make them part of its periphery. Reasoning Result: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] <||> {0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5}.'
    parser = Module_Query_Parser()
    print(parser.parse_query(query_input=input))