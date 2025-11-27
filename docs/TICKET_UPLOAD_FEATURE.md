# Ticket Upload Feature

## Overview

The ticket upload feature allows you to add new tickets to the chatbot's knowledge base dynamically. Once uploaded, these tickets become searchable and will be used by the chatbot when generating responses to customer queries.

## How It Works

1. **Upload a ticket** via JSON file or manual entry
2. **Process the ticket** - Extract subject, description, and conversation
3. **Generate searchable text** - Combine all ticket content into a searchable format
4. **Add to vector database** - Create embeddings and store in ChromaDB
5. **Use in responses** - The chatbot will retrieve and use uploaded tickets when relevant

## Usage

### Method 1: JSON File Upload

1. Click on **"📤 Upload Ticket"** expander in the main interface
2. Select **"📄 JSON File"** option
3. Upload a JSON file with ticket data (same structure as `processed_tickets.json`)
4. Preview the ticket data
5. Click **"🚀 Add to Knowledge Base"**

**JSON Format:**
```json
{
  "ticket_id": 12345,
  "subject": "Product question",
  "description": "Customer question about product",
  "status": "open",
  "priority": "normal",
  "created_at": "2025-01-15T10:00:00.000Z",
  "updated_at": "2025-01-15T10:00:00.000Z",
  "conversation": [
    {
      "author_name": "Customer Name",
      "plain_body": "Customer message text",
      "created_at": "2025-01-15T10:00:00.000Z"
    },
    {
      "author_name": "Agent Name",
      "plain_body": "Agent response text",
      "created_at": "2025-01-15T10:05:00.000Z"
    }
  ]
}
```

### Method 2: Manual Entry

1. Click on **"📤 Upload Ticket"** expander
2. Select **"✍️ Manual Entry"** option
3. Fill in ticket information:
   - **Ticket ID** (optional - auto-generated if empty)
   - **Subject** (required)
   - **Status** (open/pending/solved/closed)
   - **Priority** (optional)
   - **Created/Updated dates** (optional)
   - **Description** (optional)
   - **Conversation messages** (optional)
4. Click **"📝 Create Ticket from Manual Entry"**
5. Click **"🚀 Add to Knowledge Base"**

## Features

### Automatic Processing
- **Searchable text generation**: Automatically combines subject, description, and conversation into searchable format
- **Metadata extraction**: Extracts ticket ID, status, priority, dates, and comment count
- **Duplicate detection**: Prevents adding tickets that already exist
- **Content validation**: Ensures tickets have sufficient content (minimum 10 characters)

### Integration
- **Immediate availability**: Uploaded tickets are immediately searchable
- **Cache clearing**: Automatically clears response cache to include new tickets
- **Stats update**: Updates database statistics after upload

## Technical Details

### VectorDBManager.add_single_ticket()

```python
def add_single_ticket(self, ticket_data: Dict[str, Any]) -> bool:
    """Add a single ticket to the vector database."""
```

**Parameters:**
- `ticket_data`: Dictionary containing ticket information
  - `ticket_id` (optional): Unique ticket identifier
  - `subject` (optional): Ticket subject
  - `description` (optional): Initial description
  - `conversation` (optional): List of conversation messages
  - `status` (optional): Ticket status
  - `priority` (optional): Ticket priority
  - `created_at` (optional): Creation timestamp
  - `updated_at` (optional): Update timestamp
  - `searchable_text` (optional): Pre-generated searchable text

**Returns:**
- `True` if ticket was added successfully
- `False` if ticket already exists or has insufficient content

### Searchable Text Generation

The `_generate_searchable_text()` method creates searchable text by combining:
1. Subject (if available)
2. Description (if available)
3. All conversation messages formatted as "Author: Message"

Example output:
```
Subject: Product question

Description: Customer question about product

Customer Name: Customer message text

Agent Name: Agent response text
```

## Use Cases

1. **Add new tickets**: Upload tickets from other sources (email, chat, etc.)
2. **Test responses**: Add test tickets to see how the chatbot responds
3. **Update knowledge base**: Continuously improve the chatbot by adding new examples
4. **Temporary tickets**: Add tickets for specific campaigns or events

## Limitations

1. **Ticket ID conflicts**: If a ticket with the same ID already exists, it will be skipped
2. **Content requirements**: Tickets must have at least 10 characters of searchable content
3. **No bulk upload**: Currently supports single ticket uploads only (can upload multiple via JSON array, but only first ticket is used)
4. **No deletion**: Uploaded tickets cannot be deleted through the UI (must be done manually in ChromaDB)

## Future Enhancements

- Bulk upload support for multiple tickets
- Ticket editing and deletion
- Upload history and management
- Validation and preview before upload
- Import from CSV/Excel files
- Integration with external ticket systems (Zendesk API, etc.)

## Troubleshooting

### "Ticket already exists"
- The ticket ID is already in the database
- Solution: Use a different ticket ID or leave it empty for auto-generation

### "Insufficient content"
- The ticket doesn't have enough searchable text
- Solution: Add more content (subject, description, or conversation messages)

### "Failed to add ticket"
- Check the error message for details
- Ensure the ticket has valid data structure
- Verify the vector database is initialized

## Example Workflow

1. Customer sends an email with a question
2. Support agent copies the email content
3. Opens chatbot interface
4. Clicks "📤 Upload Ticket"
5. Selects "✍️ Manual Entry"
6. Pastes customer question as description
7. Adds agent response as conversation message
8. Clicks "🚀 Add to Knowledge Base"
9. Ticket is now available for future queries
10. When similar questions come in, chatbot will use this ticket as context





