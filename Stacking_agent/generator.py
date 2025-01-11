from typing import Union, List
from .agent import Agent
from .utils import *
from .tools import *

class ToolGenerator:
    def __init__(self):
        self.tool_mapping = {
            'Name2SMILES': 'Name2SMILES()',
            'ChemDFM': 'ChemDFM()',
            'Name2Description': 'Name2Description()',
            'Reaction2Product': 'Reaction2Product()',
        }

    def parse_tool_string(self, tool_str: str) -> List[str]:
        """解析工具字符串，返回工具名称和层级"""
        if '_' not in tool_str:
            return [tool_str, '0']
        return tool_str.split('_')

    def generate_single_tool(self, tool_spec: str) -> List[str]:
        """生成单一工具的代码行"""
        tool_name, level = self.parse_tool_string(tool_spec)
        level = int(level)
        base_name = tool_name.lower()

        code_lines = []
        # 创建基础工具
        code_lines.append(f"{base_name}_0 = {self.tool_mapping[tool_name]}")

        # 根据层级生成嵌套工具
        for i in range(1, level + 1):
            deps = [f"{base_name}_{i-1}", f"{base_name}_0"]
            deps_str = ','.join(deps)
            code_lines.append(f"{base_name}_{i} = Agent_tool(Agent([{deps_str}]))")

        return code_lines

    def generate_combined_tools(self, tools: Union[List, str]) -> List[str]:
        """生成组合工具的代码行"""
        if isinstance(tools, str):
            return self.generate_single_tool(tools)

        code_lines = []
        tool_outputs = []

        # 处理嵌套列表
        for tool in tools:
            if isinstance(tool, list):
                # 递归处理子列表
                sub_lines = self.generate_combined_tools(tool)
                code_lines.extend(sub_lines)
                # 获取最后生成的工具名作为输出
                tool_outputs.append(sub_lines[-1].split(' = ')[0])
            else:
                # 处理单个工具
                sub_lines = self.generate_single_tool(tool)
                code_lines.extend(sub_lines)
                tool_outputs.append(sub_lines[-1].split(' = ')[0])

        # 生成最终的组合工具
        if len(tool_outputs) > 1:
            combined_name = '__'.join(tool_outputs)
            deps_str = ','.join(tool_outputs)
            code_lines.append(f"final_agent = Agent_tool(Agent([{deps_str}]))")
        elif len(tool_outputs) == 1:
            code_lines.append(f"final_agent = {tool_outputs[0]}")

        return code_lines

    def generate(self, spec: Union[List, str]) -> str:
        """主要生成方法"""
        code_lines = self.generate_combined_tools(spec)
        code= '\n'.join(code_lines)
        if code.count('\n') + 1 > 2:
            wo = False
        else:
            wo = True
        # 执行生成的代码
        exec(code, globals())
        # 返回生成的 final_agent
        return final_agent,wo

       
def generate_tool(spec):
    pass

if __name__ == "__main__":
    generator = ToolGenerator()
    spec = ["ChemDFM_0"]
    code = generator.generate(spec)
    print(code)
