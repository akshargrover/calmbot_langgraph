import os
import sqlite3
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from typing import Optional, TypedDict, List
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage,ToolCall
import json
from datetime import datetime, timedelta
import calendar
import uuid
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
# Load environment variables
load_dotenv()

# --- Configuration & API Keys ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file or directly.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, temperature=0)  # Corrected class name
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key = api_key)

DB_FILE = "music_network.db"


def connect_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def retrieve_top_musicians(query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[dict]:
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        query_embedding = embeddings_model.embed_query(query)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        sql_query = "SELECT id, name, instrument, genre, skill_level, influences, city, available_online, practice_space, performance_history, description, demo_link, band_affiliations, experience_years, embedding FROM musicians WHERE embedding IS NOT NULL"
        sql_params = []

        if filters:
            if "instrument" in filters:
                sql_query += " AND instrument LIKE ?"
                sql_params.append(f"%{filters['instrument']}%")
            if "genre" in filters:
                sql_query += " AND genre LIKE ?"
                sql_params.append(f"%{filters['genre']}%")
            if "skill_level" in filters:
                sql_query += " AND skill_level = ?"
                sql_params.append(filters['skill_level'])
            if "min_experience_years" in filters:
                sql_query += " AND experience_years >= ?"
                sql_params.append(filters['min_experience_years'])
            if "city" in filters:
                sql_query += " AND city LIKE ?"
                sql_params.append(f"%{filters['city']}%")

        cursor.execute(sql_query,sql_params)
        musicians_from_db = cursor.fetchall()

        similarities = []
        for musician_row in musicians_from_db:
            db_embedding_bytes = musician_row['embedding']
            if db_embedding_bytes:
                db_embedding_np = np.frombuffer(db_embedding_bytes, dtype=np.float32)

                norm_query = np.linalg.norm(query_embedding_np)
                norm_db = np.linalg.norm(db_embedding_np)

                if norm_query == 0 or norm_db == 0:
                    similarity = -1.0
                else:
                    similarity = np.dot(query_embedding_np, db_embedding_np) / (norm_query * norm_db)
                similarities.append((similarity, musician_row))

        similarities.sort(key=lambda x: x[0], reverse=True)

        top_musicians = []
        for sim, musician_row in similarities[:top_k]:
            musician_dict = dict(musician_row)
            del musician_dict['embedding']
            musician_dict['similarity_score'] = float(sim)
            top_musicians.append(musician_dict)
       # state["retrieved_musicians"] = top_musicians
        return top_musicians

    except sqlite3.Error as e:
        print(f"Database error during musician retrieval: {e}")
        return []
    finally:
        if conn:
            conn.close()

class MoreInformationInput(BaseModel):
    musician_name:str= Field(description="Name of the musician the user wants more information about, must match one of the names from the current search results")
    retrievedmusicians:list[dict]=Field(description="List of retrieved musicians ie list of dicts containing name and id of musicians in current search results")
@tool(args_schema=MoreInformationInput)
def MoreInformation(musician_name: str, retrievedmusicians:list[dict]) -> str:
    ''' tool to display more information about the musician the user asks for '''
    # Corrected print statement to reflect the tool's name
    print(f"\n--- Tool: MoreInformation for Musician: {musician_name} ---")

    # >>> CRITICAL ISSUE: 'state' is not defined in this function's scope when called as a tool. <<<
    # >>> This line will cause a NameError at runtime. <<<
    # >>> The 'retrieved_musicians' list or the 'musician_id' should be passed as an argument. <<<
    dict3 = retrievedmusicians
    print(dict3)


    musician_id = None # Initialize musician_id
    for i in dict3:
        if i["name"] == musician_name:
            musician_id = i["id"]
            print(f"Found musician ID: {musician_id}") # Debug print
            break # Exit loop once ID is found

    if musician_id is None:
        return f"Error: Could not find Musician ID for '{musician_name}' in the retrieved list."

    conn = None
    try:
        conn = connect_db() # Assuming connect_db() is defined elsewhere and returns a connection
        cursor = conn.cursor()

        # Execute once to get the musician's info
        cursor.execute("SELECT * FROM musicians WHERE id = ?", (musician_id,))
        musician_info = cursor.fetchone() # fetchone returns a single row or None

        if not musician_info:
            return f"Sorry, I could not find detailed information for musician with ID: {musician_id}."

        # Convert the Row object to a dictionary for easier access
        musician_dict = dict(musician_info)

        # Start building the formatted string
        formatted_info = ""

        # 1. Print the Description first
        if musician_dict.get('description'):
            formatted_info += f"Description: {musician_dict['description']}\n\n"
        else:
            formatted_info += "No specific description available.\n\n"

        # 2. Add other information in a structured way
        formatted_info += f"Name: {musician_dict.get('name', 'N/A')}\n"
        formatted_info += f"Instrument: {musician_dict.get('instrument', 'N/A')}\n"
        formatted_info += f"Genre: {musician_dict.get('genre', 'N/A')}\n"
        formatted_info += f"Skill Level: {musician_dict.get('skill_level', 'N/A')}\n"
        formatted_info += f"Experience: {musician_dict.get('experience_years', 'N/A')} years\n"
        formatted_info += f"City: {musician_dict.get('city', 'N/A')}\n"

        # Availability/Practice info
        online_status = "Yes" if musician_dict.get('available_online') == 1 else "No"
        formatted_info += f"Available Online: {online_status}\n"
        formatted_info += f"Practice Space: {musician_dict.get('practice_space', 'N/A')}\n"

        # Other details
        if musician_dict.get('influences'):
            formatted_info += f"Influences: {musician_dict['influences']}\n"
        if musician_dict.get('performance_history'):
            formatted_info += f"Performance History: {musician_dict['performance_history']}\n"
        if musician_dict.get('band_affiliations'):
            formatted_info += f"Band Affiliations: {musician_dict['band_affiliations']}\n"
        if musician_dict.get('demo_link'):
            formatted_info += f"Demo Link: {musician_dict['demo_link']}\n"

        # Add a concluding remark or prompt
        formatted_info += "\nIs there anything else you'd like to know about this musician, or would you like to check their availability?"
        return formatted_info


    except sqlite3.Error as e:
        print(f"Database error in MoreInformation tool: {e}")
        return f"An error occurred while fetching more information for musician ID {musician_id}: {e}"
    finally:
        if conn:
            conn.close()




# --- Tool Definitions for LLM ---
class SelectMusicianForMeetingInput(BaseModel):
    """Selects a specific musician from the previously listed search results to schedule a meeting with."""
    musician_name: str = Field(
        description="The full name of the musician the user wants to meet with. Must exactly match one of the names from the current search results.")


@tool(args_schema=SelectMusicianForMeetingInput)
def select_musician_for_meeting(musician_name: str, retrieved_musicians: list) -> str:
    """
    Tool to select a musician by name from the current search results for scheduling a meeting.
    Retrieves retrieved_musicians as a direct argument.
    Returns the ID of the selected musician or an error message if not found.
    """
    # NO LONGER NEED `graph_state_manager` HERE!
    # current_state = graph_state_manager.get_current_state()
    # retrieved_musicians = current_state.get("retrieved_musicians")

    print(f"select_musician_for_meeting received musician_name: {musician_name}")
    print(f"select_musician_for_meeting received retrieved_musicians: {retrieved_musicians}")


    if not retrieved_musicians:
        return "No musicians have been retrieved yet. Please perform a search first."

    musician_name = str(musician_name).strip()

    for musician in retrieved_musicians:
        print(f"Checking musician in list: {musician.get('name', 'N/A')}")
        if musician.get('name', '').lower() == musician_name.lower():
            if 'id' in musician:
                return json.dumps({"musician_id": musician['id'], "musician_name": musician['name']})
                print("")
            else:
                print(f"Warning: Musician {musician.get('name')} found but has no 'id' key.")
                return f"Musician '{musician_name}' found but its ID is unavailable."

    return f"Musician '{musician_name}' not found in the current search results. Please pick one from the list."

class ModifySearchCriteriaInput(BaseModel):
    """Performs a search if It is the first search and Modifies the current search criteria to find different musicians. Can specify instrument, genre, skill level, minimum experience, or city."""
    new_query: str=Field(description="The new search query which may or may not be related to the earlier search.")

@tool(args_schema=ModifySearchCriteriaInput)
def modify_search_criteria(new_query: str) -> str:
    """
    Tool to perform a musician search from the database of musicians and answer the users query.
    The LLM should analyse the new query in relation the previous query and perform a new search when required.
    """
    musiciansearchandresponcealg(new_query)

@tool
def end_search_session() -> str:
    """
    Tool to signal the end of the current search session using state.
    """
    return "Ending session confirmed."




class ConfirmMeetingDetailsInput(BaseModel):
    """Confirms the user's name and contact information to finalize a meeting booking."""
    user_name: str = Field(description="The name of the user booking the meeting.")
    user_contact: str = Field(description="The contact number or email of the user booking the meeting.")


@tool(args_schema=ConfirmMeetingDetailsInput)
def confirm_meeting_details(user_name: str, user_contact: str) -> str:
    """
    Tool to collect and confirm user's name and contact information for a meeting.
    """
    graph_state_manager.update_state({"user_name_for_meeting": user_name, "user_contact_for_meeting": user_contact})
    return f"I have your name as '{user_name}' and contact as '{user_contact}'. Please confirm the meeting slot."


# --- TOOL 1: select_meeting_slot (renamed from display_musician_availability) ---
class SelectMeetingSlotInput(BaseModel): # RENAMED
    """
    Displays the available meeting slots for a given musician.
    """
    musician_name: str = Field(..., description="The name of the musician for whom to display availability.")
    #retrieved_musicians: list[dict] = Field(
    musician_id: str = Field(description="The ID of the musician for whom to display availability.")
@tool(args_schema=SelectMeetingSlotInput) # UPDATED args_schema
def select_meeting_slot(musician_name: str,musician_id: str) -> str: # RENAMED
    """
    Fetches and formats the availability slots for a given musician ID from the database.
    (Previously display_musician_availability)
    """
    print(f"\n--- Tool: select_meeting_slot for Musician: {musician_name} ---")
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM musicians WHERE id = ?", (musician_id,))
        musician_name_row = cursor.fetchone()
        musician_name = musician_name_row['name'] if musician_name_row else "Unknown Musician"

        cursor.execute("SELECT * FROM musician_schedules WHERE musician_id = ?", (musician_id,))
        musicianslots = cursor.fetchall()

        if not musicianslots:
            return f"Sorry, {musician_name} (ID: {musician_id}) has no availability slots listed."
        else:
            availability_text = f"\nHere are {musician_name}'s availability slots:\n"
            for j, slot in enumerate(musicianslots):
                slot_dict = dict(slot)
                is_online = slot_dict.get('is_online', 0)
                is_offline = slot_dict.get('is_offline', 0)

                mode_options = []
                if is_online == 1:
                    mode_options.append("Online")
                if is_offline == 1:
                    mode_options.append("Offline")
                mode_str = " / ".join(mode_options) if mode_options else "Unavailable"

                meeting_date = get_next_date_for_day(slot_dict.get('day_of_week'))
                if not meeting_date:
                    continue

                availability_text += (
                    f"{j + 1}. Day: {slot_dict.get('day_of_week')}, Date: {meeting_date}, "
                    f"Time: {slot_dict.get('start_time')} - {slot_dict.get('end_time')}, Mode: {mode_str}\n"
                )

            availability_text += (
                f"\nWhich slot would you like to book? Please specify the day, start time, and end time. "
                f"You can also specify 'online' or 'offline' if both options are available for a slot. "
                f"Example: 'I want to book Wednesday from 2 PM to 4 PM online.'"
            )
            return availability_text
    except sqlite3.Error as e:
        print(f"Database error in select_meeting_slot (was display_musician_availability): {e}") # UPDATED print
        return f"An error occurred while fetching availability for musician ID {musician_id}: {e}"
    finally:
        if conn:
            conn.close()

# (The book_and_confirm_meeting tool and its input model remain unchanged)

class BookAndConfirmMeetingInput(BaseModel):
    """
    Books a meeting slot for a musician and confirms details with the user.
    Requires musician ID, selected slot details, and user contact information.
    """
    musician_id: str = Field(..., description="The ID of the musician for the meeting.")
    day_of_week: str = Field(..., description="The day of the week for the meeting (e.g., 'Monday').")
    start_time: str = Field(..., description="The start time of the meeting (e.g., '10:00 AM').")
    end_time: str = Field(..., description="The end time of the meeting (e.g., '11:00 AM').")
    mode: str = Field(..., description="The meeting mode ('Online' or 'Offline').")
    user_name: str = Field(..., description="The user's full name for the booking.")
    user_contact: str = Field(..., description="The user's contact information (phone number or email).")

@tool(args_schema=BookAndConfirmMeetingInput)
def book_and_confirm_meeting(
    musician_id: str,
    day_of_week: str,
    start_time: str,
    end_time: str,
    mode: str,
    user_name: str,
    user_contact: str
) -> str:
    """
    Saves the confirmed meeting details into the database and returns a confirmation message.
    """
    print(f"\n--- Tool: book_and_confirm_meeting ---")
    conn = None
    try:
        conn = connect_db()
        cursor = conn.cursor()

        # Get musician name for the confirmation message
        cursor.execute("SELECT name FROM musicians WHERE id = ?", (musician_id,))
        musician_name_row = cursor.fetchone()
        musician_name = musician_name_row['name'] if musician_name_row else "Unknown Musician"

        # Calculate the specific meeting date
        meeting_date = get_next_date_for_day(day_of_week)
        if not meeting_date:
            return f"Error: Could not determine meeting date for {day_of_week}. Please try again."

        print(f"Attempting to book: Musician={musician_name} ({musician_id}), Date={meeting_date}, "
              f"Time={start_time}-{end_time}, Mode={mode}, User={user_name}, Contact={user_contact}")

        insert_sql = '''
                     INSERT INTO meetings(user_name, user_contact, musician_id, meeting_mode, meeting_date, \
                                          start_time, end_time, status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     '''
        rows_insert = (
            user_name,
            user_contact,
            musician_id,
            mode,
            meeting_date,
            start_time,
            end_time,
            "scheduled"
        )
        cursor.execute(insert_sql,rows_insert)
        conn.commit()


        return (f"Your meeting with {musician_name} on {meeting_date} from {start_time} to {end_time} ({mode}) ")

    except sqlite3.Error as e:
        print(f"Database error during meeting scheduling: {e}")
        return f"An error occurred while scheduling the meeting: {e}. Please try again or contact support."
    finally:
        if conn:
            conn.close()



llm_with_tools = llm.bind_tools([
    select_musician_for_meeting,
    modify_search_criteria,
    end_search_session,
    book_and_confirm_meeting,
    select_meeting_slot,
    MoreInformation
])


class GraphState(TypedDict):
    user_query: Optional[str]
    satisfied: Optional[str]
    messages: List[BaseMessage]
    retrieved_musicians: Optional[List[dict]]
    final_response_text: Optional[str]
    musicianChosen: Optional[dict]
    meetingSlot: Optional[dict]
    user_name_for_meeting: Optional[str]
    user_contact_for_meeting: Optional[str]
    tool_error: Optional[str]


class GraphStateManager:
    _state: GraphState = {
        "user_query": None,
        "satisfied": None,
        "messages": [],
        "retrieved_musicians": None,
        "final_response_text": None,
        "musicianChosen": None,
        "meetingSlot": None,
        "user_name_for_meeting": None,
        "user_contact_for_meeting": None,
        "search_filters": None,
        "tool_error": None
    }

    def update_state(self, updates: Dict[str, Any]):
        if "messages" in updates:
            new_messages = updates.pop("messages")
            for msg in new_messages:
                # Prevent adding duplicate messages, especially during internal loops
                if not any(existing_msg.content == msg.content and existing_msg.type == msg.type for existing_msg in
                           self._state["messages"]):
                    self._state["messages"].append(msg)

        for key, value in updates.items():
            self._state[key] = value

    def get_current_state(self) -> GraphState:
        return self._state


graph_state_manager = GraphStateManager()


def Takeinput(state: GraphState) -> GraphState:
    """Initial node for taking user input."""
    print("\n--- Take Input Node ---")
    user_input = input("What kind of musician are you looking for today?")
    new_state = state.copy()
    new_state["messages"].append(HumanMessage(content=user_input))
    new_state["user_query"] = user_input
    new_state["tool_error"] = None
    return new_state


def generate_musician_summary(musician: Dict[str, Any], user_query: str) -> str:
    """Generates a detailed and user-friendly summary for a musician using the LLM."""
    prompt = f"""
    You are an AI assistant helping a user find musicians. Given the following musician's details and the user's original query,
    create a concise and engaging summary for this musician, highlighting relevant aspects based on the user's query if possible.

    Musician Details:
    Name: {musician.get('name', 'N/A')}
    Instrument: {musician.get('instrument', 'N/A')}
    Genre: {musician.get('genre', 'N/A')}
    Skill Level: {musician.get('skill_level', 'N/A')}
    Experience Years: {musician.get('experience_years', 'N/A')}
    City: {musician.get('city', 'N/A')}
    Description: {musician.get('description', 'N/A')}
    Influences: {musician.get('influences', 'N/A')}
    Band Affiliations: {musician.get('band_affiliations', 'N/A')}
    Performance History: {musician.get('performance_history', 'N/A')}
    Demo Link: {musician.get('demo_link', 'N/A')}

    User's Original Query: "{user_query}"

    Please provide a summary of the musician, emphasizing details that would appeal to someone searching for them.
    Start directly with the summary, no preamble. Make it concise and engaging.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def MusicianSearchAndResponseNode(state: GraphState) -> GraphState:
    print(f"\n--- Performing Musician Search and Displaying Response ---")
    user_query = state.get("user_query", "")
    newstate=musiciansearchandresponcealg(user_query)
    return newstate

def musiciansearchandresponcealg(user_query:str):
    graph_state_manager.update_state({"search_filters": {}})  # Clear filters after use

    retrieved_musicians = retrieve_top_musicians(query=user_query)
    new_state = state.copy()
    new_state["retrieved_musicians"] = retrieved_musicians

    if not retrieved_musicians:
        final_response = "I couldn't find any musicians matching your criteria in my database. Please try a different search or be more specific."
    else:
        response_lines = [f"Here are some musicians found based on your query:{user_query}"]
        for i, musician in enumerate(retrieved_musicians):
            # Use LLM to generate a rich description
            summary = generate_musician_summary(musician, user_query)
            response_lines.append(
                f"{i + 1}. {musician['name']} (id: {musician['id']}):\n"
                f"   {summary}"  # Use the LLM-generated summary
            )
        final_response = "\n".join(response_lines)

    print(final_response)
    new_state["messages"].append(AIMessage(content=final_response))
    new_state["final_response_text"] = final_response
    new_state["satisfied"] = "awaiting_user_choice"
    return new_state


def agent_node(state: GraphState) -> GraphState:
    """
    you are a musician finder chatbot.
    you will answer questions based on the users query focusing more on the newer part but also keeping in mind context
    If the user asks for a list of musicians of any kind always call the modify_search_criteria tool.
    Node for the LLM agent to decide on the next action based on previous context (tool call or direct response)
    You have access to a database of musicians and information about them, you also have access to
    these musicians schedules and current meetings scheduled.
    If a user asks to set a meeting or schedule a meeting always call the select_musician_for_meeting tool
    and to manage getting user input when needed, only take input required for next actions
    focus mainly on the newest(last) part of the query and not the earlier part of the query that should be referred to only for context
    after the select_musician_for_meeting tool is called, agent should never respond itself it should always respond using the select_meeting_slot tool.
    """
    print("\n--- Agent Node: Deciding Next Action ---")
    new_state = state.copy()

    # Determine if a direct user input is needed
    last_message = new_state["messages"][-1] if new_state["messages"] else None

    # If the last message was an AI response that provided a final output
    # or if we explicitly set satisfied to 'awaiting_user_choice'
    # and there's no tool call from the previous AI message to execute

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # Agent just "spoke" and is waiting for user input

        user_input = input("Your action:? ")
        new_state["messages"].append(HumanMessage(content=user_input))
        new_state["user_query"] = user_input  # Update user_query with the new input

    # Now, process with the LLM (which will include the new user input if taken above)
    messages_for_llm = new_state["messages"]

    result = llm_with_tools.invoke(messages_for_llm)

    if isinstance(result, BaseMessage):
        if result.tool_calls:
            # Safely get the first tool call
            tool_call_item = result.tool_calls[0]

            # Check if it's a ToolCall object or a dictionary
            if isinstance(tool_call_item, dict):
                print(f"Agent chose to call tool1: {tool_call_item['name']}")
            elif isinstance(tool_call_item, dict):
                print(f"Agent chose to call tool: {tool_call_item.get('name', 'Unknown Tool')}")
            else:
                print(f"Agent chose to call unknown tool type: {type(tool_call_item)}")

            new_state["messages"].append(result)
            new_state["satisfied"] = "tool_called"
        elif new_state['satisfied']=='end':
            print("Thankyou for using Musician Chatbot, GOODBYE!")
        else:  # It's an AIMessage with content, but no tool_calls
            print("Agent responded directly without a tool call.")
            new_state["messages"].append(result)
            new_state["final_response_text"] = result.content
            print(f"Agent says: {result.content}")

            # Agent has responded, now it expects a new user input (handled by route to agent)
            new_state["satisfied"] = "awaiting_user_choice"  # This will route back to agent for next input cycle
    else:
        print(f"DEBUG: Unexpected LLM result type: {type(result)}. Content: {result}")
        error_message = "I'm sorry, I encountered an unexpected response. Could you please rephrase?"
        new_state["messages"].append(AIMessage(content=error_message))
        new_state["final_response_text"] = error_message
        new_state["satisfied"] = "awaiting_user_choice"  # Route back to agent to prompt

    return new_state









def tool_executor(state: GraphState) -> GraphState:
    """Execute tool calls from AI messages and update state accordingly. Ask for only those inputs required by the function """
    print("\n--- Tool Executor Node ---")
    new_state = state.copy()

    # 1. Find the last AI message with tool calls
    last_ai_message = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
            last_ai_message = msg
            break

    if not last_ai_message:
        return _handle_tool_error(new_state, "No tool calls found in recent messages")

    # 2. Get the first tool call to execute
    try:
        tool_calls = last_ai_message.tool_calls
        tool_call_item = tool_calls[0] if tool_calls else None
    except (IndexError, TypeError, AttributeError) as e:
        return _handle_tool_error(new_state, f"Invalid tool calls format: {str(e)}")

    # 3. Extract tool call details (handles both object and dict formats)
    tool_info = _extract_tool_call_info(tool_call_item)
    if not tool_info:
        return _handle_tool_error(new_state, "Could not extract tool call information")

    tool_name, tool_args, tool_call_id = tool_info

    # 4. Execute the appropriate tool
    try:
        tool_func = {
            "select_musician_for_meeting": select_musician_for_meeting,
            "book_and_confirm_meeting": book_and_confirm_meeting,
            "modify_search_criteria": modify_search_criteria,
            "end_search_session": end_search_session,
            "confirm_meeting_details": confirm_meeting_details,
            "select_meeting_slot": select_meeting_slot,
            "MoreInformation":MoreInformation
        }.get(tool_name)


        if not tool_func:
            return _handle_tool_error(new_state, f"Unknown tool: {tool_name}", tool_call_id)

        if tool_name == "select_musician_for_meeting":
            musician_name_arg = tool_args.get("musician_name")
            if musician_name_arg is None:
                return _handle_tool_error(new_state,
                                          "Missing 'musician_name' argument for select_musician_for_meeting.",
                                          tool_call_id)

            retrieved_musicians_list = new_state.get("retrieved_musicians")
            tool_output = musician_name_arg

        elif tool_name == "book_and_confirm_meeting":
            tool_output = tool_func.invoke(tool_args)

        elif tool_name=="EndSearchSession":
            new_state["satisfied"] = "end"
        else:
            tool_output = tool_func(tool_args)  # Or tool_func.invoke(tool_args) if they are LangChain Runnables


        new_state["messages"].append(ToolMessage(
            content=tool_output,
            tool_call_id=tool_call_id
        ))

        new_state = _update_state_from_tool_output(new_state, tool_output,tool_name)


    except Exception as e:
        return _handle_tool_error(
            new_state,
            f"Error executing {tool_name}: {str(e)}",
            tool_call_id
        )

    return new_state


def _extract_tool_call_info(tool_call_item) -> tuple:
    """Extract (tool_name, tool_args, tool_call_id) from tool call item."""
    tool_name = None
    tool_args = {}
    tool_call_id = str(uuid.uuid4())  # Default ID

    # Handle object-style tool calls (ToolCall instance)
    if hasattr(tool_call_item, 'name'):
        tool_name = getattr(tool_call_item, 'name')
        tool_args = getattr(tool_call_item, 'args', {})
        tool_call_id = getattr(tool_call_item, 'id', tool_call_id)
    # Handle dictionary-style tool calls
    elif isinstance(tool_call_item, dict):
        tool_name = tool_call_item.get('name')
        tool_args = tool_call_item.get('args', {})
        tool_call_id = tool_call_item.get('id', tool_call_id)

    if not tool_name:
        return None

    return (tool_name, tool_args, tool_call_id)

def _update_state_from_tool_output(state: GraphState, tool_output: str, tool_name:str) -> GraphState:
    """Update graph state based on tool execution output."""
    new_state = state.copy()
        # Check for specific JSON output from select_musician_for_meeting

    if "musician_id" in tool_output and "musician_name" in tool_output:
        try:
            musician_info = json.loads(tool_output)
            new_state["musicianChosen"] = {
                "id": musician_info["musician_id"],
                "name": musician_info["musician_name"]
            }

            new_state["satisfied"] = "awaiting_user_choice" # After selecting musician, LLM should display options.
        except (json.JSONDecodeError, KeyError):
            new_state["satisfied"] = "awaiting_user_choice" # Fallback if parsing fails

    if tool_name=="end_search_session":
        new_state["satisfied"] = "end"

    else:
        new_state["satisfied"] = "awaiting_user_choice"

    return new_state


def _handle_tool_error(state: GraphState, error_msg: str, tool_call_id: str = None) -> GraphState:
    """Handle tool execution errors consistently."""
    if not tool_call_id:
        tool_call_id = str(uuid.uuid4())

    print(f"TOOL ERROR: {error_msg}")
    state["messages"].append(ToolMessage(
        content=error_msg,
        tool_call_id=tool_call_id
    ))
    state["tool_error"] = error_msg
    state["satisfied"] = "awaiting_user_choice"
    return state

def get_next_date_for_day(day_name: str) -> str:
    today = datetime.today()
    today_idx = today.weekday()
    try:
        target_idx = list(calendar.day_name).index(day_name)
    except ValueError:
        print(f"Warning: Invalid day name '{day_name}'. Cannot determine next date.")
        return ""

    days_ahead = (target_idx - today_idx + 7) % 7
    if days_ahead == 0:
        days_ahead = 7

    next_date = today + timedelta(days=days_ahead)
    return next_date.strftime("%Y-%m-%d")

def end_conversation_node(state: GraphState) -> GraphState:
    print("\n--- Ending Conversation ---")
    state["satisfied"] = "end"
    print(state["satisfied"])
    return state


# --- Routing Logic ---
def route_state(state: GraphState) -> str:
    satisfied_choice = state.get("satisfied")
    print(satisfied_choice)
    if satisfied_choice == "end":
        return "end_conversation"
    elif satisfied_choice == "new_search_required":
        print("ROUTE: Detected 'new_search_required', routing to 'musician_search'")
        return "new_search_required"
    elif satisfied_choice == "tool_called":
        print("ROUTE: Detected 'tool_called', routing to 'tool_executor'")
        return "tool_called"
    elif satisfied_choice == "awaiting_user_choice":
        print("ROUTE: Detected 'awaiting_user_choice', routing back to 'agent'")
        return "awaiting_user_choice"
        # Route back to agent for its internal input handling
    else:
        print(f"DEBUG: Unexpected satisfied_choice in router: {satisfied_choice}. Defaulting to 'agent'")
        return "agent"


# --- Graph Building ---
workflow = StateGraph(GraphState)

workflow.add_node("take_input", Takeinput)
workflow.add_node("musician_search", MusicianSearchAndResponseNode)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_executor", tool_executor)
workflow.add_node("end_conversation", end_conversation_node)

workflow.set_entry_point("take_input")
workflow.add_edge("take_input", "agent")
#workflow.add_edge("musician_search", "agent")

workflow.add_conditional_edges(
    "agent",
    route_state,
    {
        "tool_called": "tool_executor",
        "new_search_required": "musician_search",
        "end_conversation": "end_conversation",
        "awaiting_user_choice": "agent",  # Loop back to agent for its internal input handling

    }
)

workflow.add_edge("tool_executor", "agent")
workflow.add_edge("end_conversation", END)

app = workflow.compile()

initial_graph_state: GraphState = {
    "user_query": None,
    "satisfied": "awaiting_user_choice",
    "messages": [],
    "retrieved_musicians": None,
    "final_response_text": None,
    "musicianChosen": None,
    "meetingSlot": None,
    "user_name_for_meeting": None,
    "user_contact_for_meeting": None,
    "search_filters": {},
    "tool_error": None
}

# --- FastAPI Setup ---
app1 = FastAPI()
app1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    state: dict = None  # Optional, for multi-turn

class ChatResponse(BaseModel):
    response: str
    state: dict

@app1.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Use the provided state or start with initial_graph_state
    user_state = req.state or initial_graph_state.copy()
    user_state = user_state.copy()  # Defensive copy
    user_state["user_query"] = req.message
    user_state["messages"].append(HumanMessage(content=req.message))
    # Run the graph for one turn (simulate a single chat step)
    new_state = app_graph.invoke(user_state)
    # Find the latest AI message for the response
    ai_msgs = [m for m in new_state["messages"] if isinstance(m, AIMessage)]
    response = ai_msgs[-1].content if ai_msgs else new_state.get("final_response_text", "...")
    return ChatResponse(response=response, state=new_state)

# --- Graph for API (single turn) ---
app_graph = workflow.compile()

if __name__ == "__main__":
    import sys
    if "runserver" in sys.argv:
        uvicorn.run("main:app1", host="0.0.0.0", port=8000, reload=True)
    else:
        print("Welcome to the Musician Network! How can I help you find a musician today?")
        print(f"Registered nodes: {workflow.nodes.keys()}")
        print(f"Routing to: {route_state(initial_graph_state)}")
        state = initial_graph_state.copy()
        app.invoke(state)
