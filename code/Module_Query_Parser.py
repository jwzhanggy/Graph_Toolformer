import re
import time


class Module_Query_Parser():

    def __init__(self, name='Method_Query_Parser', description='query parser'):
        self.name = name
        self.description = description

    def parse_function_calls(self, input_string):
        function_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^\]]*)\)'
        function_calls = re.findall(function_call_pattern, input_string)

        parsed_function_calls = []
        for function_call in function_calls:
            function_name, parameters_str = function_call
            parameter_list = []
            nested_count = 0
            nested_function_call_str = ''
            for parameter in parameters_str.split(','):
                parameter = parameter.strip()
                if '(' in parameter or ')' in parameter:
                    count_left_parentheses = parameter.count('(')
                    count_right_parentheses = parameter.count(')')
                    nested_count = nested_count + count_left_parentheses - count_right_parentheses
                    nested_function_call_str += ','+parameter
                    if nested_count == 0:
                        parameter_list.append(self.parse_function_calls(nested_function_call_str))
                        nested_function_call_str = ''
                else:
                    if nested_count == 0:
                        parameter_list.append(parameter)
                    else:
                        nested_function_call_str += ','+parameter
            parsed_function_calls.append((function_name, parameter_list))

        return parsed_function_calls

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
    input = 'Output: Nodes [ GR ( GL ( "dodecahedral_graph", GB("a", GS("c"))), "periphery" )  -->  r ] have the largest eccentricity [GR(GL("dodecahedral_graph"), "eccentricity")] in the dodecahedral graph, which make them part of its periphery.'
    parser = Module_Query_Parser()
    print(parser.parse_query(query_input=input))