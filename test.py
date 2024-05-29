from function_call.main import AnthropicFunctionCalling


def main():
    afc = AnthropicFunctionCalling()
    messages = []
    dataframes = []

    prompt = "What are some statistics for Klay Thompson?"

    messages.append({"role": "user", "content": prompt})

    response = afc.client.beta.tools.messages.create(
        model=afc.MODEL_NAME,
        max_tokens=4096,
        tools=afc.tools,
        messages=messages
    )

    initial_response_cost = afc.calculate_cost(response.model, response)

    messages.append({"role": "assistant", "content": response.content})

    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool_result = afc.process_tool_call(tool_name, tool_input)

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(tool_result),
                }
            ],
        }
        )

        response = afc.client.beta.tools.messages.create(
            model=afc.MODEL_NAME,
            max_tokens=4096,
            tools=afc.tools,
            messages=messages
        )

        second_response_cost = afc.calculate_cost(response.model, response)

        messages.append({"role": "assistant", "content": response.content})

        final_response = next(
            (block.text for block in response.content if hasattr(block, "text")),
            None,
        )

        formatted_response = f"""
        Based on the response, ensure to format the message as follows:

        {final_response}

        Rules:

        - The message should be formatted as a string.
        - The message should be a response to the user's question.
        - The message should not mention the tool used.
        - Do not include the xml tags in the response
        - Respond in the exact format in the examples below

        Example:

        <get_player_info>"Draymond Greene currently plays for the Golden State Warriors."</get_player_info>
        <get_player_statistics>"Here are some statistics regarding Draymond Greene:
        1. The most amount of points he scored in a season was 100 and that occured during the 2020-2021 season.
        2. His free throw percentage was 100% during the same season.
        3. For the season he had his most points scored, he played for the Golden State Warriors.
        "</get_player_statistics>
        <get_league_titles>"The Golden State Warriors have won 6 league titles."</get_league_titles>
        """

        messages.append({"role": "user", "content": final_response})

        formatted_response = [{'role': 'user', 'content': formatted_response}]

        final_output_response = afc.client.messages.create(
            model='claude-3-opus-20240229',
            max_tokens=4096,
            temperature=0.0,
            messages=formatted_response
        )
        final_output_response_text = final_output_response.content[0].text
        final_output_response_cost = afc.calculate_cost(response.model, final_output_response)
        total_cost = initial_response_cost + second_response_cost + final_output_response_cost

    messages.append({"role": "assistant", "content": final_output_response_text})
    dataframes.append((prompt, total_cost))

if __name__ == "__main__":
    main()