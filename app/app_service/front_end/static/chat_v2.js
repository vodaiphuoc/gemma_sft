function display_chat_history(topic_value) {
    /**
     * Given `topic_value` of clicked `tag_id` element, send post request to server
     * and render old history chat
     * Args:
     *  - topic_value (str)
     */

    fetch('/load_history', {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            topic_value: topic_value
        })
    })
    .then((response)=> {
        if (response.ok) {
            response.json()
            .then((chat_history)=>{
                for (i = 0; i < chat_history.length; i++) {
                    showMessageByRole(
                        chat_history[i]['parts'],
                        chat_history[i]['role'],
                        chat_history[i]['timestamp']
                    );
                }
            });
        } else {
            console.log('error loading old messages');
        }
        
    });

}


class WebSocket_Handler {
    /**
     * Class for holding values of `WebSocket` instance and `topic_value`
     * Assign an event handler when recieve data from FastAPI server
     */

    constructor() {
        this.ws = new WebSocket("ws://localhost:8080/ws");
        this.topic_value = '';
        
        this.ws.addEventListener("message", (event) => {
            let processed_data = JSON.parse(event.data);
            
            // display new message
            showMessageByRole(processed_data['msg'], 'model');
            
            // key `topic` only appear when `new_chat` is created
            if ('topic' in processed_data) {
                let new_id = $("[id^=topic_]").length;
                $("#all_topics").append(`<li id="topic_${new_id}">${processed_data['topic']}</li>`);
    
                this.topic_value = processed_data['topic'];
            }
        });
    }
   
}

function chatsession_callback() {
    /**
     * Callback function when a history topic or `create_new_chat` are 
     * clicked:
     *  (1) determine topic, if `create_new_chat`, topic is empty string
     *  (2) assign new value to `topic_value` of `WebSocket_Handler` instance
     *  (3) render chat history of current topic
     */

    if (this.id === 'create_new_chat') {
        ws_handler.topic_value = ''
        // if a new chat is clicked, empty .messages
        clear_current_messages();

    } else {
        ws_handler.topic_value = $(`#${this.id}`).text();
        // step (2)
        display_chat_history(ws_handler.topic_value);
    }
    
}

function load_topics() {
    /**
     * Load all topics when start papges
     * assign `chatsession_callback` to each `#topic_*` li tag
     */

    fetch('/load_topics', {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            'user': 'user'
        })
    })
    .then((response)=> {
        if (response.ok) {
            response.json()
            .then((response_topics)=>{

                for (i = 0; i < response_topics.length; i++) {
                    // append topic elements to `all_topics`
                    $("#all_topics").append(`<li id="topic_${i}">${response_topics[i]}</li>`)
                    
                    // assign click handler
                    $(`li#topic_${i}`).on("click", chatsession_callback);
                    console.log('done assign handler to topics');
                }

                return null;
            });
        } else {
            console.log('error loading topics');
        }
        return null;
    });

    return null;
}


/**-------------Main Process here------------------------- */
const ws_handler = new WebSocket_Handler();

/**
 * (1) Load all old topics
 * (2) Assign event handler to `create_new_chat`
 * (3) Sends a message when the 'Enter' key is pressed
 * (4) Assign event handler to `send_button`
 */
$(document).ready(function(){
    // (1)
    load_topics(ws_handler);

    // (2)
    $("#create_new_chat").on('click', chatsession_callback);
    
    // (3)
    $('#msg_input').keydown(function(e) {
        // Check for 'Enter' key
        if (e.key === 'Enter') {
            // Prevent default behaviour of enter key
            e.preventDefault();
			// Trigger send button click event
            $('#send_button').click();
        }
    });

    // (4)
    $("#send_button").click(function() {
        // step (3)
        var user_input_message = $('#msg_input').val();
        let user_timestamp = (getCurrentTimestamp()).toLocaleString('en-IN', {
            month: 'short',
            day: 'numeric',
            hour: 'numeric',
            minute: 'numeric',
        });
        showMessageByRole(user_input_message, 'user',user_timestamp);
        $('#msg_input').val('');

        ws_handler.ws.send(JSON.stringify({
            topic: ws_handler.topic_value,
            user_message: user_input_message,
            time_stamp: user_timestamp
        }));
    });
})

