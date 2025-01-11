from rxn4chemistry import RXN4ChemistryWrapper


class Reaction2Product:
    name: str = "Reaction2Product"
    description: str = "Input only one molecule name, returns SMILES. Note: the results returned by this tool may not necessarily be correct."
    def __init__(self,**tool_args):
        self.api_key ='apk-7b38266d6a3399478d1d66bab30070752734f51fe79dd0ff018c6232eeffeb9fe435bc8a9f151803bdda64fb8c73b7145cdded4347184c4584d81167b12c4ec96ecf2b533a7060cc54ea9a41dc45f2d1'
        self.rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=self.api_key)
        self.rxn4chemistry_wrapper.create_project('test_wrapper')
    
    def _run(self, query: str,**tool_args) -> str:

        try:
            response = self.rxn4chemistry_wrapper.predict_reaction(
                query
            )
            results = self.rxn4chemistry_wrapper.get_predict_reaction_results(
                response['prediction_id']
            )
            return results['response']['payload']['attempts'][0]['smiles'].split('>>')[1]
        except:
            return "Incorrect input :One possible cause is that the input is incorrect, please modify your input."
    
    def __str__(self):
        return "Reaction2Product tool"

    def __repr__(self):
        return self.__str__()

    def wo_run(self,query,debug=False):
        answer = self._run(query.split('\nChemical reaction equation:')[1].split('>>')[0])
        if answer == "Incorrect input :One possible cause is that the input is incorrect, please modify your input.":
            return ""
        return answer


if __name__ == '__main__':
    print(Reaction2Product()._run('C1CCOC1.CC(=O)[O-].CC(=O)[O-].CCOC(C)=O.COc1cccc([Mg+])c1.O.O=C1c2ccc(OS(=O)(=O)C(F)(F)F)cc2C(=O)N1Cc1cccnc1.[Br-].[Cl-].[Cl-].[Pd+2].[Zn+2]'))