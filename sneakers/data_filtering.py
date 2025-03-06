import pandas as pd
import json
import os
from typing import Dict, Optional
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from prompts import DATA_ANALYSIS_PROMPT_TEMPLATE


class DataFieldMapping(BaseModel):
    """데이터 필드 매핑을 위한 Pydantic 모델"""

    field_mappings: Dict[str, str] = Field(description="원본 데이터 필드와 대상 형식 필드 간의 매핑")
    missing_fields: Dict[str, Optional[str]] = Field(
        description="형식에는 있지만 원본 데이터에 없는 필드에 대한 추정 값"
    )


class DataFilter:
    """
    다양한 형식의 데이터 파일을 로드하고 정제하는 클래스
    """

    def __init__(self, format_file_path: str = "format.json"):
        """
        DataFilter 클래스 초기화

        Args:
            format_file_path: 필드 형식이 정의된 JSON 파일 경로
        """
        self.format_file_path = format_file_path
        self.format_fields = self._load_format_fields()

    def _load_format_fields(self) -> Dict:
        """
        format.json 파일에서 필드 정보를 로드

        Returns:
            필드 정보가 담긴 딕셔너리
        """
        try:
            with open(self.format_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"경고: {self.format_file_path} 파일을 찾을 수 없습니다.")
            return {}
        except json.JSONDecodeError:
            print(f"경고: {self.format_file_path} 파일이 올바른 JSON 형식이 아닙니다.")
            return {}

    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        파일을 로드하고 DataFrame으로 변환

        Args:
            file_path: 로드할 파일 경로

        Returns:
            변환된 DataFrame 또는 오류 시 None
        """
        _, ext = os.path.splitext(file_path)

        try:
            if ext.lower() == ".json":
                return self._load_json(file_path)
            elif ext.lower() == ".csv":
                return self._load_csv(file_path)
            else:
                print(f"지원하지 않는 파일 형식입니다: {ext}")
                return None
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {e}")
            return None

    def _load_json(self, file_path: str) -> pd.DataFrame:
        """
        JSON 파일을 로드하고 DataFrame으로 변환

        Args:
            file_path: JSON 파일 경로

        Returns:
            변환된 DataFrame
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 데이터가 리스트가 아니면 리스트로 변환
        if not isinstance(data, list):
            data = [data]

        df = pd.DataFrame(data)
        return self._apply_format(df)

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        CSV 파일을 로드하고 DataFrame으로 변환

        Args:
            file_path: CSV 파일 경로

        Returns:
            변환된 DataFrame
        """
        df = pd.read_csv(file_path)
        return self._apply_format(df)

    def _apply_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        format.json에 정의된 필드 형식을 DataFrame에 적용

        Args:
            df: 원본 DataFrame

        Returns:
            형식이 적용된 DataFrame
        """
        if not self.format_fields:
            return df

        # 필요한 칼럼만 선택
        available_fields = [
            field for field in self.format_fields if field in df.columns
        ]

        # 필드가 없으면 원본 반환
        if not available_fields:
            print("경고: format.json에 정의된 필드가 데이터에 없습니다.")
            return df

        # 필드 순서대로 DataFrame 재구성
        result_df = df[available_fields].copy()

        # 누락된 필드 추가
        for field in self.format_fields:
            if field not in result_df.columns:
                result_df[field] = None

        # format.json에 정의된 순서대로 칼럼 재정렬
        return result_df[self.format_fields]

    def save_file(
        self, df: pd.DataFrame, output_path: str, file_format: str = "csv"
    ) -> bool:
        """
        DataFrame을 파일로 저장

        Args:
            df: 저장할 DataFrame
            output_path: 저장 경로
            file_format: 저장 형식 ('csv' 또는 'json')

        Returns:
            저장 성공 여부
        """
        try:
            if file_format.lower() == "csv":
                df.to_csv(output_path, index=False)
            elif file_format.lower() == "json":
                df.to_json(output_path, orient="records", force_ascii=False, indent=2)
            else:
                print(f"지원하지 않는 저장 형식입니다: {file_format}")
                return False
            return True
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return False

    def ai_analyze_and_format(
        self, df: pd.DataFrame, model_name: str = "gpt-4o-mini", temperature: float = 0.0
    ) -> pd.DataFrame:
        """
        AI를 사용하여 데이터셋을 분석하고 format에 맞게 정제

        Args:
            df: 원본 DataFrame
            model_name: 사용할 AI 모델 이름 ('gpt-4o-mini', 'o3-mini', 'claude-3-opus', 'claude-3-sonnet' 등)
            temperature: 모델의 temperature 값

        Returns:
            정제된 DataFrame
        """
        if not self.format_fields:
            print("경고: format.json이 비어있어 AI 분석을 진행할 수 없습니다.")
            return df

        # 샘플 데이터 준비 (처음 5개 행)
        sample_data = df.head(5).to_dict(orient="records")

        # 출력 파서 설정
        parser = PydanticOutputParser(pydantic_object=DataFieldMapping)

        # LLM 모델 설정
        if model_name.startswith("gpt"):
            llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        elif model_name.startswith("claude"):
            llm = ChatAnthropic(model_name=model_name, temperature=temperature)
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

        # 프롬프트 템플릿 설정
        prompt = PromptTemplate(
            template=DATA_ANALYSIS_PROMPT_TEMPLATE,
            input_variables=["sample_data", "format_fields"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # LangChain 체인 생성
        chain = LLMChain(llm=llm, prompt=prompt)

        # 체인 실행
        try:
            result = chain.run(
                sample_data=json.dumps(sample_data, ensure_ascii=False, indent=2),
                format_fields=json.dumps(
                    self.format_fields, ensure_ascii=False, indent=2
                ),
            )

            # 결과 파싱
            parsed_output = parser.parse(result)
            field_mappings = parsed_output.field_mappings
            missing_fields = parsed_output.missing_fields

            # 결과 적용
            result_df = pd.DataFrame()

            # 매핑된 필드 적용
            for target_field, source_field in field_mappings.items():
                if source_field in df.columns:
                    result_df[target_field] = df[source_field]

            # 누락된 필드에 추정값 적용
            for field, value in missing_fields.items():
                if field not in result_df.columns:
                    result_df[field] = value

            # format.json에 정의된 순서대로 칼럼 재정렬
            result_df = result_df.reindex(columns=self.format_fields)

            return result_df

        except Exception as e:
            print(f"AI 분석 중 오류 발생: {e}")
            return self._apply_format(df)  # 오류 시 기본 포맷 적용


# 사용 예시
if __name__ == "__main__":
    # 데이터 필터 객체 생성
    data_filter = DataFilter("format.json")

    # 파일 로드 및 변환
    df = data_filter.load_file("data.csv")
    if df is not None:
        print("데이터 로드 성공:")
        print(df.head())

        # AI를 사용하여 데이터 분석 및 정제
        ai_df = data_filter.ai_analyze_and_format(df, model_name="gpt-4o-mini")
        print("\nAI 분석 결과:")
        print(ai_df.head())

        # 변환된 데이터 저장
        data_filter.save_file(ai_df, "ai_filtered_data.csv")
