import re

class Module_Query_Parser():

    def __init__(self, name='Method_Query_Parser', description='query parser'):
        self.name = name
        self.description = description

    def parse_function_calls(self, input_string):
        function_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)'
        function_calls = re.findall(function_call_pattern, input_string)

        parsed_function_calls = []

        for function_call in function_calls:
            function_name, parameters_str = function_call
            parameter_list = []
            for parameter in parameters_str.split(','):
                if '(' in parameter and ')' in parameter:
                    parameter_list.append(self.parse_function_calls(parameter))
                else:
                    parameter_list.append(parameter)
            parsed_function_calls.append((function_name, parameter_list))

        return parsed_function_calls

    def parse_output_variable(self, input_string):
        output_variable_pattern = r'-->([a-zA-Z_][a-zA-Z0-9_]*)'
        output_variable = re.search(output_variable_pattern, input_string)
        return output_variable.group(1) if output_variable else None

    def parse_query(self, query_text_input):
        function_parameters = self.parse_function_calls(query_text_input)
        output_variable = self.parse_output_variable(query_text_input)
        return function_parameters, output_variable